#!/usr/bin/env python3
"""load_gen.py — drive main_server with concurrent streaming clients.

Closed loop (N clients, each sends the next request when its previous one
ends) or open loop (Poisson arrivals at a rate). Measures, per request, time to
first token, per-token latency and total latency, and overall generated
tokens/s; reads /metrics before and after so KV blocks and cancellations are
checked server-side. --cancel-fraction closes that share of streams after
--cancel-after tokens, to test that cancelled requests free their blocks.

Standard library only (no aiohttp): one thread per in-flight request, plain
sockets through http.client, SSE parsed line by line.

Usage:
  tools/load_gen.py --url http://gpu:8080 --requests 64 --concurrency 16 --max-tokens 48
  tools/load_gen.py --url http://gpu:8080 --requests 64 --rate 8            # Poisson, 8 req/s
  tools/load_gen.py ... --cancel-fraction 0.5 --cancel-after 8 --json bench/stage9_x.json
"""

import argparse
import http.client
import json
import random
import statistics
import threading
import time
from urllib.parse import urlparse

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 + 1 =",
    "Once upon a time, in a village by the sea,",
    "The three laws of thermodynamics are",
]


def one_request(url, prompt, max_tokens, temperature, cancel_after):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stream": True})
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft = None
    tokens = 0
    cancelled = False
    finish = None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        ev = json.loads(payload)
        choice = ev["choices"][0]
        if choice.get("delta", {}).get("text", "") != "" or choice.get("finish_reason"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens += 1 if choice.get("finish_reason") is None else 0
            finish = choice.get("finish_reason") or finish
        if cancel_after and tokens >= cancel_after:
            conn.close()          # client goes away mid-stream
            cancelled = True
            break
    t1 = time.perf_counter()
    conn.close()
    return {"ttft": ttft or (t1 - t0), "latency": t1 - t0, "tokens": tokens, "cancelled": cancelled, "finish": finish}


def metrics(url):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("GET", "/metrics")
    m = json.loads(conn.getresponse().read())
    conn.close()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16, help="closed loop: in-flight clients")
    ap.add_argument("--rate", type=float, default=0.0, help="open loop: Poisson arrivals per second (0 = closed loop)")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cancel-fraction", type=float, default=0.0)
    ap.add_argument("--cancel-after", type=int, default=8)
    ap.add_argument("--json", help="write a bench record here")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rng = random.Random(1)
    plan = []
    for i in range(args.requests):
        plan.append({"prompt": PROMPTS[i % len(PROMPTS)],
                     "max_tokens": args.max_tokens // 2 + (i * 7) % (args.max_tokens + 1),
                     "cancel_after": args.cancel_after if rng.random() < args.cancel_fraction else 0})

    before = metrics(args.url)
    results = [None] * len(plan)
    lock = threading.Lock()
    t_start = time.perf_counter()

    def worker(i):
        r = one_request(args.url, plan[i]["prompt"], plan[i]["max_tokens"], args.temperature, plan[i]["cancel_after"])
        r["submitted"] = submit_t[i]
        with lock:
            results[i] = r

    submit_t = [0.0] * len(plan)
    threads = []
    if args.rate > 0:
        t = 0.0
        for i in range(len(plan)):
            t += rng.expovariate(args.rate)
            now = time.perf_counter() - t_start
            if t > now:
                time.sleep(t - now)
            submit_t[i] = time.perf_counter() - t_start
            th = threading.Thread(target=worker, args=(i,)); th.start(); threads.append(th)
    else:
        sem = threading.Semaphore(args.concurrency)
        def gated(i):
            with sem:
                submit_t[i] = time.perf_counter() - t_start
                worker(i)
        for i in range(len(plan)):
            th = threading.Thread(target=gated, args=(i,)); th.start(); threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t_start
    time.sleep(0.3)   # let the last cancellations retire
    after = metrics(args.url)

    done = [r for r in results if r and not r["cancelled"]]
    canc = [r for r in results if r and r["cancelled"]]
    tokens = sum(r["tokens"] for r in results if r)
    pct = lambda xs, p: (sorted(xs)[min(len(xs) - 1, round(p * (len(xs) - 1)))] * 1000) if xs else 0.0
    ttft = [r["ttft"] for r in results if r]
    lat = [r["latency"] for r in done]
    tok_lat = [(r["latency"] - r["ttft"]) / max(1, r["tokens"] - 1) for r in done if r["tokens"] > 1]
    summary = {
        "label": args.label, "requests": len(plan), "completed": len(done), "cancelled": len(canc),
        "mode": f"poisson {args.rate}/s" if args.rate > 0 else f"closed loop x{args.concurrency}",
        "wall_s": wall, "tokens": tokens, "tok_s": tokens / wall,
        "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95),
        "latency_p50_ms": pct(lat, 0.5), "latency_p95_ms": pct(lat, 0.95),
        "token_latency_mean_ms": statistics.mean(tok_lat) * 1000 if tok_lat else 0.0,
        "server_before": before, "server_after": after,
        "kv_blocks_in_use_after": after["kv_blocks_in_use"],
        "server_cancellations": after["cancellations"] - before["cancellations"],
    }
    print(f"{summary['mode']}: {len(done)} done, {len(canc)} cancelled, {tokens} tokens in {wall:.1f} s = "
          f"{summary['tok_s']:.1f} tok/s | ttft p50 {summary['ttft_p50_ms']:.0f} ms p95 {summary['ttft_p95_ms']:.0f} | "
          f"latency p50 {summary['latency_p50_ms']:.0f} p95 {summary['latency_p95_ms']:.0f} | "
          f"token {summary['token_latency_mean_ms']:.1f} ms | server: blocks in use {after['kv_blocks_in_use']}, "
          f"cancellations {summary['server_cancellations']}, preemptions {after['preemptions'] - before['preemptions']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
