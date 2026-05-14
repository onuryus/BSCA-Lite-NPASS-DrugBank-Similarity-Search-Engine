#pragma once
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace bscs {

// Split a tab-delimited line into fields. Reuses the output vector.
inline void split_tsv(std::string_view line, std::vector<std::string_view>& out) {
    out.clear();
    size_t start = 0;
    while (true) {
        size_t pos = line.find('\t', start);
        if (pos == std::string_view::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
}

inline std::vector<std::string_view> split_tsv(std::string_view line) {
    std::vector<std::string_view> out;
    split_tsv(line, out);
    return out;
}

// Extract first tab-separated field (the SMILES column in Enamine).
inline std::string_view first_field(std::string_view line) {
    size_t pos = line.find('\t');
    return pos == std::string_view::npos ? line : line.substr(0, pos);
}

// Strip CXSMILES extension: "CC1=CC=CC=C1 |...|" → "CC1=CC=CC=C1"
inline std::string strip_cxsmiles(std::string_view s) {
    size_t sp = s.find(' ');
    if (sp != std::string_view::npos) s = s.substr(0, sp);
    return std::string(s);
}

// Trim trailing whitespace/CR.
inline std::string_view trim_right(std::string_view s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\t'))
        s.remove_suffix(1);
    return s;
}

// Parse a header line into a column-name → index map.
inline std::vector<std::string> parse_header(std::string_view header_line) {
    auto fields = split_tsv(trim_right(header_line));
    std::vector<std::string> names;
    names.reserve(fields.size());
    for (auto& f : fields) names.emplace_back(f);
    return names;
}

// Find column index by name (case-sensitive). Returns -1 if not found.
inline int col_index(const std::vector<std::string>& names, const std::string& name) {
    for (int i = 0; i < (int)names.size(); i++)
        if (names[i] == name) return i;
    return -1;
}

// Write serialized string: uint16_t len + chars. Returns bytes written.
inline void write_str(std::FILE* f, const std::string& s) {
    uint16_t len = static_cast<uint16_t>(std::min(s.size(), (size_t)65535));
    std::fwrite(&len, 2, 1, f);
    std::fwrite(s.data(), 1, len, f);
}

inline std::string read_str(std::FILE* f) {
    uint16_t len = 0;
    if (std::fread(&len, 2, 1, f) != 1) return {};
    std::string s(len, '\0');
    if (len) std::fread(s.data(), 1, len, f);
    return s;
}

} // namespace bscs
