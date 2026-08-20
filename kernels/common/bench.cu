#include "bench.cuh"
#include "cuda_check.cuh"

#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

namespace bench {

namespace {

// NVML is initialized lazily once. Under WSL2 NVML works for reads (clocks,
// temp, driver version) even though nvidia-smi writes are passthrough-blocked.
bool nvml_ready() {
    static bool inited = false;
    static bool ok = false;
    if (!inited) {
        inited = true;
        ok = (nvmlInit_v2() == NVML_SUCCESS);
        if (!ok) std::fprintf(stderr, "bench: NVML unavailable; clocks/temp will read 0\n");
    }
    return ok;
}

std::string exec_capture(const char* cmd) {
    std::string out;
    if (FILE* p = popen(cmd, "r")) {
        char buf[256];
        while (fgets(buf, sizeof(buf), p)) out += buf;
        pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

} // namespace

GpuState query_gpu_state() {
    GpuState s;
    if (!nvml_ready()) return s;
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2(0, &dev) != NVML_SUCCESS) return s;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &s.sm_clock_mhz);
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &s.mem_clock_mhz);
    nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &s.temp_c);
    char drv[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    if (nvmlSystemGetDriverVersion(drv, sizeof(drv)) == NVML_SUCCESS) s.driver_version = drv;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) s.gpu_name = name;
    return s;
}

Result run(const std::string& kernel, const std::string& dims,
           const std::function<void()>& launch, int warmup, int iters,
           const std::string& gemm_path) {
    Result r;
    r.kernel = kernel;
    r.dims = dims;
    r.gemm_path = gemm_path;
    r.warmup_iters = warmup;
    r.timed_iters = iters;

    for (int i = 0; i < warmup; i++) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    r.before = query_gpu_state();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
    }

    r.after = query_gpu_state();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times_ms.begin(), times_ms.end());
    r.min_ms = times_ms.front();
    r.max_ms = times_ms.back();
    r.median_ms = (iters % 2) ? times_ms[iters / 2]
                              : 0.5 * (times_ms[iters / 2 - 1] + times_ms[iters / 2]);

    // A locked card should hold its SM clock across the run. Small jitter is
    // normal; >2% drift means the lock didn't take or thermals moved it.
    if (r.before.sm_clock_mhz && r.after.sm_clock_mhz) {
        double drift = std::abs((double)r.after.sm_clock_mhz - r.before.sm_clock_mhz) /
                       r.before.sm_clock_mhz;
        if (drift > 0.02) {
            r.clock_stable = false;
            std::fprintf(stderr,
                         "bench: WARNING SM clock drifted %u -> %u MHz during '%s %s' — "
                         "lock not holding\n",
                         r.before.sm_clock_mhz, r.after.sm_clock_mhz, kernel.c_str(),
                         dims.c_str());
        }
    }
    return r;
}

std::string write_record(const Result& r, const std::string& out_dir) {
    std::string commit = exec_capture("git rev-parse HEAD 2>/dev/null");
    bool dirty = system("git diff --quiet 2>/dev/null") != 0;
    if (dirty) {
        std::fprintf(stderr,
                     "bench: WARNING writing record from a DIRTY tree — commit hash is "
                     "not reproducible (stamped dirty:true)\n");
    }

    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&now));

    std::ostringstream path;
    path << out_dir << "/" << r.kernel << "_" << r.dims << "_" << ts << ".json";

    std::ofstream f(path.str());
    f << "{\n"
      << "  \"kernel\": \"" << r.kernel << "\",\n"
      << "  \"dims\": \"" << r.dims << "\",\n"
      << "  \"gemm_path\": \"" << r.gemm_path << "\",\n"
      << "  \"median_ms\": " << r.median_ms << ",\n"
      << "  \"min_ms\": " << r.min_ms << ",\n"
      << "  \"max_ms\": " << r.max_ms << ",\n"
      << "  \"warmup_iters\": " << r.warmup_iters << ",\n"
      << "  \"timed_iters\": " << r.timed_iters << ",\n"
      << "  \"commit\": \"" << commit << "\",\n"
      << "  \"dirty\": " << (dirty ? "true" : "false") << ",\n"
      << "  \"gpu\": \"" << r.before.gpu_name << "\",\n"
      << "  \"driver\": \"" << r.before.driver_version << "\",\n"
      << "  \"cuda_toolkit\": " << CUDART_VERSION << ",\n"
      << "  \"sm_clock_mhz_before\": " << r.before.sm_clock_mhz << ",\n"
      << "  \"sm_clock_mhz_after\": " << r.after.sm_clock_mhz << ",\n"
      << "  \"mem_clock_mhz\": " << r.before.mem_clock_mhz << ",\n"
      << "  \"temp_c_before\": " << r.before.temp_c << ",\n"
      << "  \"temp_c_after\": " << r.after.temp_c << ",\n"
      << "  \"clock_stable\": " << (r.clock_stable ? "true" : "false") << ",\n"
      << "  \"model\": null\n"
      << "}\n";
    return path.str();
}

} // namespace bench
