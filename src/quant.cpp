// quant.cpp — see quant.h.

#include "quant.h"
#include "f16.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../third_party/nlohmann/json.hpp"

namespace llm {

using nlohmann::json;

const char* qkind_name(QKind k) {
    return k == QKind::kInt8 ? "int8" : k == QKind::kInt4 ? "int4" : "bf16";
}

QTensor QMatrix::view() const {
    QTensor t;
    t.kind = kind; t.rows = rows; t.cols = cols;
    t.q = q.data(); t.scales = scales.data(); t.zeros = zeros.empty() ? nullptr : zeros.data();
    return t;
}

QMatrix quantize(const uint16_t* w, int64_t rows, int64_t cols, QKind kind) {
    if (kind != QKind::kInt8 && kind != QKind::kInt4) throw std::runtime_error("quantize: kind");
    if (kind == QKind::kInt4 && cols % 2) throw std::runtime_error("quantize: int4 needs even cols");
    QMatrix m;
    m.kind = kind; m.rows = rows; m.cols = cols;
    const int64_t G = (cols + kGroup - 1) / kGroup;
    m.scales.assign(size_t(rows) * G, 0);
    m.q.assign(kind == QKind::kInt4 ? size_t(rows) * cols / 2 : size_t(rows) * cols, 0);
    if (kind == QKind::kInt4) m.zeros.assign(size_t(rows) * G, 0);
    std::vector<float> row(static_cast<size_t>(cols), 0.0f);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) row[size_t(c)] = f16::bf16_to_f32(w[r * cols + c]);
        for (int64_t g = 0; g < G; g++) {
            const int64_t c0 = g * kGroup, c1 = std::min(cols, c0 + kGroup);
            if (kind == QKind::kInt8) {
                float amax = 0.0f;
                for (int64_t c = c0; c < c1; c++) amax = std::max(amax, std::fabs(row[size_t(c)]));
                // The scale is stored as bf16: quantise against the STORED value so
                // dequantisation uses exactly the scale the kernel will see.
                const uint16_t sb = f16::f32_to_bf16(amax > 0 ? amax / 127.0f : 1.0f);
                const float s = f16::bf16_to_f32(sb);
                m.scales[size_t(r * G + g)] = sb;
                for (int64_t c = c0; c < c1; c++) {
                    int v = int(std::lrint(row[size_t(c)] / s));
                    v = std::max(-127, std::min(127, v));
                    m.q[size_t(r * cols + c)] = uint8_t(int8_t(v));
                }
            } else {
                // The range must include zero: otherwise the zero point falls
                // outside 0..15, gets clamped, and a group of same-sign values
                // is crushed (measured: 98% error on a two-element group).
                float lo = 0.0f, hi = 0.0f;
                for (int64_t c = c0; c < c1; c++) { lo = std::min(lo, row[size_t(c)]); hi = std::max(hi, row[size_t(c)]); }
                const uint16_t sb = f16::f32_to_bf16(hi > lo ? (hi - lo) / 15.0f : 1.0f);
                const float s = f16::bf16_to_f32(sb);
                int z = int(std::lrint(-lo / s));
                z = std::max(0, std::min(15, z));
                m.scales[size_t(r * G + g)] = sb;
                m.zeros[size_t(r * G + g)] = uint8_t(z);
                for (int64_t c = c0; c < c1; c++) {
                    int v = int(std::lrint(row[size_t(c)] / s)) + z;
                    v = std::max(0, std::min(15, v));
                    uint8_t& byte = m.q[size_t((r * cols + c) / 2)];
                    if (c % 2 == 0) byte = uint8_t((byte & 0xF0) | v);
                    else byte = uint8_t((byte & 0x0F) | (v << 4));
                }
            }
        }
    }
    return m;
}

float dequant(const QTensor& t, int64_t r, int64_t c) {
    const int64_t G = t.groups(), g = c / kGroup;
    const float s = f16::bf16_to_f32(t.scales[r * G + g]);
    if (t.kind == QKind::kInt8) return s * float(int8_t(t.q[r * t.cols + c]));
    const uint8_t byte = t.q[(r * t.cols + c) / 2];
    const int v = (c % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    return s * float(v - int(t.zeros[r * G + g]));
}

// ---------------------------------------------------------------- writer
void QuantWriter::add(const std::string& name, const QMatrix& m) {
    entries_.push_back({name, m.kind, {m.rows, m.cols}, m.q, m.zeros, m.scales});
}
void QuantWriter::add_bf16(const std::string& name, const uint16_t* data, std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    Entry e{name, QKind::kBf16, shape, {}, {}, {}};
    e.scales.assign(data, data + n);   // bf16 payload rides in `scales`
    entries_.push_back(std::move(e));
}
void QuantWriter::write(const std::string& path) const {
    json h = json::object();
    size_t off = 0;
    auto span = [&](size_t n) { json r = json::array({off, off + n}); off += n; return r; };
    for (const Entry& e : entries_) {
        json j{{"dtype", qkind_name(e.kind)}, {"shape", e.shape}, {"group", kGroup}};
        if (e.kind == QKind::kBf16) {
            j["data"] = span(e.scales.size() * 2);
        } else {
            j["data"] = span(e.q.size());
            j["scales"] = span(e.scales.size() * 2);
            if (!e.zeros.empty()) j["zeros"] = span(e.zeros.size());
        }
        h[e.name] = j;
    }
    const std::string hs = h.dump();
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write("LLMQ0001", 8);
    const uint64_t hl = hs.size();
    f.write(reinterpret_cast<const char*>(&hl), 8);
    f.write(hs.data(), std::streamsize(hs.size()));
    for (const Entry& e : entries_) {
        if (e.kind == QKind::kBf16) { f.write(reinterpret_cast<const char*>(e.scales.data()), std::streamsize(e.scales.size() * 2)); continue; }
        f.write(reinterpret_cast<const char*>(e.q.data()), std::streamsize(e.q.size()));
        f.write(reinterpret_cast<const char*>(e.scales.data()), std::streamsize(e.scales.size() * 2));
        if (!e.zeros.empty()) f.write(reinterpret_cast<const char*>(e.zeros.data()), std::streamsize(e.zeros.size()));
    }
}

// ---------------------------------------------------------------- reader
QuantFile::~QuantFile() { if (map_) munmap(map_, map_size_); }

void QuantFile::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat " + path); }
    map_size_ = size_t(st.st_size);
    map_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap " + path); }
    const auto* base = static_cast<const uint8_t*>(map_);
    if (map_size_ < 16 || std::memcmp(base, "LLMQ0001", 8) != 0) throw std::runtime_error("not an .llmq file: " + path);
    uint64_t hl = 0;
    std::memcpy(&hl, base + 8, 8);
    if (16 + hl > map_size_) throw std::runtime_error("truncated header: " + path);
    const json h = json::parse(base + 16, base + 16 + hl);
    const uint8_t* blob = base + 16 + hl;
    const size_t blob_size = map_size_ - 16 - hl;
    auto at = [&](const json& r) {
        const size_t b = r[0].get<size_t>(), e = r[1].get<size_t>();
        if (e > blob_size || b > e) throw std::runtime_error("bad range in " + path);
        return blob + b;
    };
    for (auto it = h.begin(); it != h.end(); ++it) {
        const json& j = it.value();
        QTensor t;
        const std::string d = j.at("dtype").get<std::string>();
        t.kind = d == "int8" ? QKind::kInt8 : d == "int4" ? QKind::kInt4 : QKind::kBf16;
        const auto shape = j.at("shape").get<std::vector<int64_t>>();
        t.rows = shape.size() >= 1 ? shape[0] : 0;
        t.cols = shape.size() >= 2 ? shape[1] : 1;
        if (t.kind == QKind::kBf16) {
            t.bf16 = reinterpret_cast<const uint16_t*>(at(j.at("data")));
        } else {
            if (j.value("group", kGroup) != kGroup) throw std::runtime_error("group size mismatch in " + path);
            t.q = at(j.at("data"));
            t.scales = reinterpret_cast<const uint16_t*>(at(j.at("scales")));
            if (j.contains("zeros")) t.zeros = at(j.at("zeros"));
            kind_ = t.kind;
        }
        tensors_[it.key()] = t;
    }
}

const QTensor& QuantFile::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not in .llmq: " + name);
    return it->second;
}
std::vector<std::string> QuantFile::names() const {
    std::vector<std::string> v;
    for (auto& [k, _] : tensors_) v.push_back(k);
    return v;
}

// ---------------------------------------------------------------- model
std::string QuantModel::file_for(const std::string& dir, QKind kind) {
    return dir + (kind == QKind::kInt4 ? "/model.q4.llmq" : "/model.q8.llmq");
}

void QuantModel::load(const std::string& dir, QKind k) {
    cfg = ModelConfig::from_file(dir + "/config.json");
    kind = k;
    file_.open(file_for(dir, k));
    auto mat = [&](const std::string& n, int64_t rows, int64_t cols) {
        const QTensor& t = file_.get(n);
        if (t.rows != rows || t.cols != cols || t.kind != k)
            throw std::runtime_error("shape/kind mismatch for " + n);
        return t;
    };
    auto vec = [&](const std::string& n, int64_t len) {
        const QTensor& t = file_.get(n);
        if (t.kind != QKind::kBf16 || t.rows != len) throw std::runtime_error("bad bf16 tensor " + n);
        return t;
    };
    const int64_t H = cfg.hidden_size, q_out = cfg.num_attention_heads * cfg.head_dim,
                  kv_out = cfg.num_key_value_heads * cfg.head_dim, I = cfg.intermediate_size;
    embed_tokens = mat("model.embed_tokens.weight", cfg.vocab_size, H);
    if (!cfg.tie_word_embeddings) lm_head = mat("lm_head.weight", cfg.vocab_size, H);
    final_norm = vec("model.norm.weight", H);
    layers.resize(size_t(cfg.num_hidden_layers));
    for (int64_t i = 0; i < cfg.num_hidden_layers; i++) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        QLayer& L = layers[size_t(i)];
        L.q_w = mat(p + "self_attn.q_proj.weight", q_out, H);
        L.k_w = mat(p + "self_attn.k_proj.weight", kv_out, H);
        L.v_w = mat(p + "self_attn.v_proj.weight", kv_out, H);
        L.o_w = mat(p + "self_attn.o_proj.weight", H, q_out);
        L.gate_w = mat(p + "mlp.gate_proj.weight", I, H);
        L.up_w = mat(p + "mlp.up_proj.weight", I, H);
        L.down_w = mat(p + "mlp.down_proj.weight", H, I);
        L.input_ln = vec(p + "input_layernorm.weight", H);
        L.post_attn_ln = vec(p + "post_attention_layernorm.weight", H);
        L.has_bias = cfg.attention_bias;
        if (L.has_bias) {
            L.q_b = vec(p + "self_attn.q_proj.bias", q_out);
            L.k_b = vec(p + "self_attn.k_proj.bias", kv_out);
            L.v_b = vec(p + "self_attn.v_proj.bias", kv_out);
        }
    }
}

} // namespace llm
