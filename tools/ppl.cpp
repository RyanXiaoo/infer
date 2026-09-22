// ppl.cpp — perplexity of the engine on the eval tokens (Stage 10).
//
// Teacher-forced: the tokens are fed as a prompt in windows of `window`
// tokens; for every position the engine returns -log p(next token) computed
// from its own logits on the device (GpuBatch::score). Perplexity =
// exp(mean nll). Same code path for bf16 and quantised weights, so the number
// isolates the effect of quantisation.
//
// Usage: ppl [window=512] [max_tokens=8192]   (model: LLM_MODEL, LLM_QUANT)

#include "../kernels/batch_gpu.h"
#include "../kernels/model_gpu.h"
#include "../src/model.h"
#include "../src/model_select.h"
#include "../src/npy.h"
#include "../src/quant.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const int64_t window = argc > 1 ? std::atoll(argv[1]) : 512;
    const int64_t max_tokens = argc > 2 ? std::atoll(argv[2]) : 8192;
    const std::string root = MODEL_ROOT;
    auto arr = llm::npy::load_npy(llm::golden_dir(root) + "/eval_tokens.npy");
    std::vector<int64_t> ids(arr.i64(), arr.i64() + std::min<int64_t>(arr.numel(), max_tokens));

    llm::Model model;
    llm::QuantModel qmodel;
    const std::string quant = llm::quant_name();
    if (quant.empty()) model.load(llm::model_dir(root));
    else qmodel.load(llm::model_dir(root), quant == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8);
    std::unique_ptr<llm::GpuModel> gpu = quant.empty() ? std::make_unique<llm::GpuModel>(model)
                                                       : std::make_unique<llm::GpuModel>(qmodel);
    llm::GpuBatch batch(*gpu, 1, window + 1, llm::GemmPath::kMine, 0, false, false);

    double nll_sum = 0.0;
    int64_t n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t w0 = 0; w0 + 1 < int64_t(ids.size()); w0 += window) {
        const int64_t w1 = std::min<int64_t>(int64_t(ids.size()), w0 + window + 1);
        std::vector<int64_t> chunk(ids.begin() + w0, ids.begin() + w1);
        std::vector<float> nll = batch.score(0, chunk);   // nll[i] for target chunk[i+1]
        for (float v : nll) { nll_sum += v; n++; }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("model %s quant=%s: %lld tokens, window %lld: perplexity %.4f (mean nll %.4f), %.1f s\n",
                llm::model_name().c_str(), quant.empty() ? "bf16" : quant.c_str(), (long long)n,
                (long long)window, std::exp(nll_sum / double(n)), nll_sum / double(n), s);
    return 0;
}
