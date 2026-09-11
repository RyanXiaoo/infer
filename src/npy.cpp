#include "npy.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace llm::npy {

int64_t Array::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("npy: " + msg); }

// Parse one .npy stream at `p` of size `n` bytes.
// Format: \x93NUMPY, 1-byte major, 1-byte minor, header-len (2 bytes v1 / 4 bytes v2),
// then a Python-dict literal: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
Array parse_npy(const uint8_t* p, size_t n) {
    if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) fail("bad magic");
    uint8_t major = p[6];
    size_t header_len, header_off;
    if (major == 1) {
        uint16_t hl;
        std::memcpy(&hl, p + 8, 2);
        header_len = hl;
        header_off = 10;
    } else {
        uint32_t hl;
        std::memcpy(&hl, p + 8, 4);
        header_len = hl;
        header_off = 12;
    }
    if (header_off + header_len > n) fail("header exceeds buffer");
    std::string h(reinterpret_cast<const char*>(p) + header_off, header_len);

    auto extract = [&](const std::string& key) -> std::string {
        size_t k = h.find("'" + key + "'");
        if (k == std::string::npos) fail("missing key " + key);
        size_t colon = h.find(':', k);
        size_t start = h.find_first_not_of(" ", colon + 1);
        // value ends at the comma that closes this entry (tuples contain commas,
        // so for shape we scan to the matching ')')
        size_t end;
        if (h[start] == '(') {
            end = h.find(')', start) + 1;
        } else {
            end = h.find_first_of(",}", start);
        }
        return h.substr(start, end - start);
    };

    Array a;
    std::string descr = extract("descr");
    if (descr.find("<f4") != std::string::npos) a.dtype = Dtype::F32;
    else if (descr.find("<u2") != std::string::npos) a.dtype = Dtype::U16;
    else if (descr.find("<i8") != std::string::npos) a.dtype = Dtype::I64;
    else fail("unsupported descr " + descr);

    if (extract("fortran_order").find("False") == std::string::npos)
        fail("fortran_order must be False (C-order only)");

    std::string shape = extract("shape");   // e.g. "(2, 3)" or "(5,)" or "()"
    for (size_t i = 0; i < shape.size();) {
        if (isdigit(shape[i])) {
            size_t j = i;
            while (j < shape.size() && isdigit(shape[j])) j++;
            a.shape.push_back(std::stoll(shape.substr(i, j - i)));
            i = j;
        } else {
            i++;
        }
    }

    size_t itemsize = a.dtype == Dtype::F32 ? 4 : a.dtype == Dtype::U16 ? 2 : 8;
    size_t nbytes = static_cast<size_t>(a.numel()) * itemsize;
    if (header_off + header_len + nbytes > n) fail("data exceeds buffer");
    a.data.assign(p + header_off + header_len, p + header_off + header_len + nbytes);
    return a;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

} // namespace

Array load_npy(const std::string& path) {
    auto buf = read_file(path);
    return parse_npy(buf.data(), buf.size());
}

// Minimal zip walker for np.savez output: sequence of local-file records,
// each 30-byte header + name + (extra) + STORED (uncompressed) data.
std::map<std::string, Array> load_npz(const std::string& path) {
    auto buf = read_file(path);
    std::map<std::string, Array> out;
    size_t pos = 0;
    while (pos + 30 <= buf.size()) {
        uint32_t sig;
        std::memcpy(&sig, &buf[pos], 4);
        if (sig != 0x04034b50) break;   // not a local file header: central directory reached

        uint16_t method, name_len, extra_len;
        uint32_t comp_size32, uncomp_size32;
        std::memcpy(&method, &buf[pos + 8], 2);
        std::memcpy(&comp_size32, &buf[pos + 18], 4);
        std::memcpy(&uncomp_size32, &buf[pos + 22], 4);
        std::memcpy(&name_len, &buf[pos + 26], 2);
        std::memcpy(&extra_len, &buf[pos + 28], 2);

        std::string name(reinterpret_cast<const char*>(&buf[pos + 30]), name_len);
        size_t data_off = pos + 30 + name_len + extra_len;

        // numpy writes zip64 entries (force_zip64): 32-bit size fields hold the
        // 0xFFFFFFFF sentinel and the real 64-bit sizes live in the extra field,
        // record id 0x0001, fields in order (uncompressed, compressed) — only
        // the ones that were saturated are present.
        uint64_t comp_size = comp_size32, uncomp_size = uncomp_size32;
        if (comp_size32 == 0xFFFFFFFFu || uncomp_size32 == 0xFFFFFFFFu) {
            size_t ep = pos + 30 + name_len;
            size_t eend = ep + extra_len;
            bool found = false;
            while (ep + 4 <= eend) {
                uint16_t id, dlen;
                std::memcpy(&id, &buf[ep], 2);
                std::memcpy(&dlen, &buf[ep + 2], 2);
                if (id == 0x0001) {
                    size_t fp = ep + 4;
                    if (uncomp_size32 == 0xFFFFFFFFu) { std::memcpy(&uncomp_size, &buf[fp], 8); fp += 8; }
                    if (comp_size32 == 0xFFFFFFFFu) { std::memcpy(&comp_size, &buf[fp], 8); }
                    found = true;
                    break;
                }
                ep += 4 + dlen;
            }
            if (!found) fail("zip64 sentinel without zip64 extra field in " + name);
        }

        if (method != 0) fail("npz member is compressed; use np.savez, not savez_compressed");
        if (data_off + comp_size > buf.size()) fail("zip member exceeds file");

        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            out[name.substr(0, name.size() - 4)] = parse_npy(&buf[data_off], comp_size);
        }
        pos = data_off + comp_size;
    }
    if (out.empty()) fail("no npy members found in " + path);
    return out;
}

} // namespace llm::npy
