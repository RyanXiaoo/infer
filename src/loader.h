// loader.h — safetensors checkpoint loader.
//
// A .safetensors file is: 8 bytes little-endian header length N, then N bytes of
// JSON mapping tensor name -> {dtype, shape, data_offsets}, then the raw tensor
// data. data_offsets are relative to the START OF THE DATA SECTION (byte 8+N),
// not the file — the classic parse bug is forgetting that.
//
// The file is mmap'd read-only and tensors are zero-copy views into it: weights
// stay bf16 in memory (converting to fp32 on load would double memory for
// nothing — Stage 3 uploads bf16 to the GPU anyway; CPU math converts at
// use-time via f16.h).
//
// Plain C++, no CUDA (boundary rule).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

enum class Dtype { BF16, F16, F32, I64, UNKNOWN };

const char* dtype_name(Dtype d);
size_t dtype_size(Dtype d);

struct Tensor {
    std::string name;
    Dtype dtype = Dtype::UNKNOWN;
    std::vector<int64_t> shape;
    const void* data = nullptr;   // points into the mmap; valid while the file is open
    size_t nbytes = 0;

    int64_t numel() const;
    // Typed accessors; caller asserts dtype. bf16/f16 surface as uint16_t bits.
    const uint16_t* u16() const { return static_cast<const uint16_t*>(data); }
    const float* f32() const { return static_cast<const float*>(data); }
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Throws std::runtime_error with a specific message on any malformed input.
    void open(const std::string& path);

    bool has(const std::string& name) const;
    const Tensor& get(const std::string& name) const;   // throws if absent
    std::vector<std::string> names() const;             // sorted

    size_t data_section_size() const { return data_size_; }
    // For the offset-sanity test: each tensor's [begin, end) within the data section.
    std::pair<size_t, size_t> data_offsets(const std::string& name) const;

private:
    void* map_ = nullptr;
    size_t map_size_ = 0;
    const uint8_t* data_base_ = nullptr;   // start of the data section
    size_t data_size_ = 0;
    std::vector<Tensor> tensors_;                       // sorted by name
    std::vector<std::pair<size_t, size_t>> offsets_;    // parallel to tensors_
    int find(const std::string& name) const;
};

} // namespace llm
