#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "../third_party/nlohmann/json.hpp"

using nlohmann::json;

namespace llm {

const char* dtype_name(Dtype d) {
    switch (d) {
        case Dtype::BF16: return "BF16";
        case Dtype::F16: return "F16";
        case Dtype::F32: return "F32";
        case Dtype::I64: return "I64";
        default: return "UNKNOWN";
    }
}

size_t dtype_size(Dtype d) {
    switch (d) {
        case Dtype::BF16:
        case Dtype::F16: return 2;
        case Dtype::F32: return 4;
        case Dtype::I64: return 8;
        default: return 0;
    }
}

static Dtype parse_dtype(const std::string& s) {
    if (s == "BF16") return Dtype::BF16;
    if (s == "F16") return Dtype::F16;
    if (s == "F32") return Dtype::F32;
    if (s == "I64") return Dtype::I64;
    return Dtype::UNKNOWN;
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

SafetensorsFile::~SafetensorsFile() {
    if (map_) munmap(map_, map_size_);
}

void SafetensorsFile::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("fstat failed: " + path); }
    map_size_ = static_cast<size_t>(st.st_size);
    if (map_size_ < 8) { close(fd); throw std::runtime_error("file too small: " + path); }

    map_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);   // mmap keeps its own reference
    if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap failed: " + path); }

    const uint8_t* base = static_cast<const uint8_t*>(map_);

    // 8-byte little-endian header length.
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    if (8 + header_len > map_size_) throw std::runtime_error("header length exceeds file");

    // Data section starts at 8 + header_len; all data_offsets are relative to it.
    data_base_ = base + 8 + header_len;
    data_size_ = map_size_ - 8 - header_len;

    json header = json::parse(base + 8, base + 8 + header_len);

    for (auto& [name, entry] : header.items()) {
        if (name == "__metadata__") continue;   // header metadata, not a tensor

        Tensor t;
        t.name = name;
        t.dtype = parse_dtype(entry.at("dtype").get<std::string>());
        if (t.dtype == Dtype::UNKNOWN)
            throw std::runtime_error("unsupported dtype for " + name + ": " +
                                     entry.at("dtype").get<std::string>());
        t.shape = entry.at("shape").get<std::vector<int64_t>>();

        auto off = entry.at("data_offsets").get<std::vector<size_t>>();
        if (off.size() != 2 || off[1] < off[0] || off[1] > data_size_)
            throw std::runtime_error("bad data_offsets for " + name);
        size_t expect = static_cast<size_t>(t.numel()) * dtype_size(t.dtype);
        if (off[1] - off[0] != expect)
            throw std::runtime_error("offset span != numel*itemsize for " + name);

        t.data = data_base_ + off[0];
        t.nbytes = off[1] - off[0];
        tensors_.push_back(std::move(t));
        offsets_.emplace_back(off[0], off[1]);
    }

    // Sort by name (with offsets_ kept parallel) for binary search in find().
    std::vector<size_t> order(tensors_.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return tensors_[a].name < tensors_[b].name; });
    std::vector<Tensor> ts;
    std::vector<std::pair<size_t, size_t>> os;
    ts.reserve(order.size());
    os.reserve(order.size());
    for (size_t i : order) {
        ts.push_back(std::move(tensors_[i]));
        os.push_back(offsets_[i]);
    }
    tensors_ = std::move(ts);
    offsets_ = std::move(os);
}

int SafetensorsFile::find(const std::string& name) const {
    auto it = std::lower_bound(tensors_.begin(), tensors_.end(), name,
                               [](const Tensor& t, const std::string& n) { return t.name < n; });
    if (it == tensors_.end() || it->name != name) return -1;
    return static_cast<int>(it - tensors_.begin());
}

bool SafetensorsFile::has(const std::string& name) const { return find(name) >= 0; }

const Tensor& SafetensorsFile::get(const std::string& name) const {
    int i = find(name);
    if (i < 0) throw std::runtime_error("tensor not found: " + name);
    return tensors_[i];
}

std::vector<std::string> SafetensorsFile::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& t : tensors_) out.push_back(t.name);
    return out;
}

std::pair<size_t, size_t> SafetensorsFile::data_offsets(const std::string& name) const {
    int i = find(name);
    if (i < 0) throw std::runtime_error("tensor not found: " + name);
    return offsets_[i];
}

} // namespace llm
