// model_select.h — which checkpoint the tools and tests run against.
//
// Portability rule (plans/roadmap.md): the engine never names a model; every
// dimension comes from config.json. This header is the one place the HARNESS
// picks a checkpoint, so a second model can run the same golden ladder:
//
//   LLM_MODEL=Qwen2.5-1.5B-Instruct ./build/test_forward_gpu
//
// Unset -> the 0.5B dev model, whose goldens live directly in tests/golden/.
// Any other model's goldens live in tests/golden/<name>/ (tools/dump_logits.py
// --model <name> writes them there).

#pragma once

#include <cstdlib>
#include <string>

namespace llm {

inline const char* kDefaultModel = "Qwen2.5-0.5B-Instruct";

inline std::string model_name() {
    const char* env = std::getenv("LLM_MODEL");
    return env && env[0] ? env : kDefaultModel;
}
inline std::string model_dir(const std::string& root) {
    return root + "/models/" + model_name();
}
inline std::string golden_dir(const std::string& root) {
    const std::string name = model_name();
    return name == kDefaultModel ? root + "/tests/golden" : root + "/tests/golden/" + name;
}

} // namespace llm
