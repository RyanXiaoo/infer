// quant.h — weight quantization: group-wise int8 / int4 (Stage 10).
//
// Decode is bound by weight bytes per token, so the weights are stored with
// fewer bits and dequantised in registers on the way into the multiply. Along
// each row of a [out x in] matrix, groups of kGroup consecutive weights (the
// direction the GEMV streams) share one bf16 scale:
//   int8, symmetric:  q = round(w / s),        s = absmax / 127,       w ~ s * q
//   int4, asymmetric: q = round(w / s) + z,    s = (max - min) / 15,   w ~ s * (q - z)
//                     with the range [min, max] widened to include 0 (so z is in 0..15)
//                     z = round(-min / s), q and z in 0..15, two per byte, low nibble first.
// Round-to-nearest only (no calibration); the accuracy cost is measured, not
// assumed (tools/ppl).
//
// Container (.llmq): 8-byte magic "LLMQ0001", u64 header length, JSON header
// {name: {dtype, shape, group, data, scales?, zeros?}} with [begin, end) byte
// ranges into the blob that follows. Norms and biases are stored as bf16
// tensors in the same file, so a quantised model is one file. mmap-read.
//
// Plain C++ (boundary rule): the quantiser, the round-trip reference and the
// reader are Mac-testable; the GPU kernels live in kernels/ops/quant.cu.

#pragma once

#include "loader.h"
#include "model_config.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llm {

constexpr int kGroup = 128;

enum class QKind { kBf16, kInt8, kInt4 };
const char* qkind_name(QKind k);

// One quantised [rows x cols] matrix, as views into a buffer or an mmap.
struct QTensor {
    QKind kind = QKind::kBf16;
    int64_t rows = 0, cols = 0;
    const uint8_t* q = nullptr;        // int8: rows*cols bytes; int4: rows*cols/2
    const uint16_t* scales = nullptr;  // bf16, rows * groups
    const uint8_t* zeros = nullptr;    // int4 only, rows * groups (one per byte)
    const uint16_t* bf16 = nullptr;    // kBf16 only (norms, biases)
    int64_t groups() const { return (cols + kGroup - 1) / kGroup; }
    size_t q_bytes() const { return kind == QKind::kInt4 ? size_t(rows) * cols / 2 : size_t(rows) * cols; }
};

// Owning quantised matrix (quantiser output / test fixture).
struct QMatrix {
    QKind kind = QKind::kInt8;
    int64_t rows = 0, cols = 0;
    std::vector<uint8_t> q, zeros;
    std::vector<uint16_t> scales;
    QTensor view() const;
};

// Quantise a bf16 [rows x cols] matrix (cols % 2 == 0 for int4).
QMatrix quantize(const uint16_t* w_bf16, int64_t rows, int64_t cols, QKind kind);
// Reference dequantisation of one element (what the kernels must reproduce).
float dequant(const QTensor& t, int64_t r, int64_t c);

// Writer: collects tensors then writes one .llmq file.
class QuantWriter {
public:
    void add(const std::string& name, const QMatrix& m);
    void add_bf16(const std::string& name, const uint16_t* data, std::vector<int64_t> shape);
    void write(const std::string& path) const;
private:
    struct Entry { std::string name; QKind kind; std::vector<int64_t> shape; std::vector<uint8_t> q, zeros; std::vector<uint16_t> scales; };
    std::vector<Entry> entries_;
};

// Reader: mmap + header; tensors are views valid while the file is open.
class QuantFile {
public:
    QuantFile() = default;
    ~QuantFile();
    QuantFile(const QuantFile&) = delete;
    QuantFile& operator=(const QuantFile&) = delete;
    void open(const std::string& path);
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const QTensor& get(const std::string& name) const;
    std::vector<std::string> names() const;
    QKind kind() const { return kind_; }   // the matrices' kind
private:
    void* map_ = nullptr;
    size_t map_size_ = 0;
    std::map<std::string, QTensor> tensors_;
    QKind kind_ = QKind::kBf16;
};

// A model whose matrices come from a .llmq file. Norms/biases are bf16 views
// (QTensor::bf16). Same wiring rules as Model (names from config + layers).
struct QLayer {
    QTensor q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
    QTensor q_b, k_b, v_b;        // kBf16; rows == 0 when absent
    QTensor input_ln, post_attn_ln;
    bool has_bias = false;
};
struct QuantModel {
    ModelConfig cfg;
    QTensor embed_tokens;         // quantised [vocab x hidden]; tied LM head
    QTensor final_norm;
    std::vector<QLayer> layers;
    QKind kind = QKind::kInt8;
    // Loads <model_dir>/config.json + <model_dir>/model.<q8|q4>.llmq.
    void load(const std::string& model_dir, QKind kind);
    static std::string file_for(const std::string& model_dir, QKind kind);
private:
    QuantFile file_;
};

} // namespace llm
