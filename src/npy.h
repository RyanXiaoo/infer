// npy.h — minimal reader for the numpy formats the golden files use.
//
// Supports exactly the subset tools/dump_logits.py produces:
//   .npy: little-endian, C-order, dtypes <f4 (float32), <u2 (uint16), <i8 (int64)
//   .npz: an UNCOMPRESSED zip (np.savez, not savez_compressed) of .npy members
//
// Written at Stage 1 on purpose: Stage 2's parity tests need to load goldens
// from C++, and discovering that mid-debug would be the worst time.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llm::npy {

enum class Dtype { F32, U16, I64 };

struct Array {
    Dtype dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;   // owned copy, little-endian C-order

    int64_t numel() const;
    const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
    const uint16_t* u16() const { return reinterpret_cast<const uint16_t*>(data.data()); }
    const int64_t* i64() const { return reinterpret_cast<const int64_t*>(data.data()); }
};

// Both throw std::runtime_error with a specific message on malformed input.
Array load_npy(const std::string& path);
// Keys are member names without the ".npy" suffix (numpy's convention).
std::map<std::string, Array> load_npz(const std::string& path);

} // namespace llm::npy
