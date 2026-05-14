#pragma once
// shard_config.hpp — JSON-serializable sharding metadata for build_sharded_database /
// search_sharded_database.  Each shard is an independent FAISS IVFPQ index covering
// a contiguous range [mol_start, mol_end) of the global molecule list.
// FAISS vector IDs equal the global molecule index, so a single EnamineReader
// (with the full offset file) serves all shards without any CXSMILES duplication.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bscs {

struct ShardEntry {
    int         id        = 0;
    uint64_t    mol_start = 0;
    uint64_t    mol_end   = 0;
    std::string faiss_path;
};

struct ShardConfig {
    uint64_t    total_molecules = 0;
    std::string cxsmiles_path;
    std::string offsets_path;
    std::string header_path;
    int         fp_radius  = 3;
    int         fp_nbits   = 1024;
    int         nlist      = 4096;
    int         pq_m       = 16;
    int         pq_nbits   = 8;
    std::vector<ShardEntry> shards;

    // ── JSON write ────────────────────────────────────────────────────────────
    void save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("ShardConfig: cannot write " + path);
        auto esc = [](const std::string& s) -> std::string {
            std::string r;
            for (char c : s) {
                if (c == '"')  r += "\\\"";
                else if (c == '\\') r += "\\\\";
                else r += c;
            }
            return r;
        };
        out << "{\n";
        out << "  \"total_molecules\": " << total_molecules << ",\n";
        out << "  \"cxsmiles_path\": \""  << esc(cxsmiles_path) << "\",\n";
        out << "  \"offsets_path\": \""   << esc(offsets_path)  << "\",\n";
        out << "  \"header_path\": \""    << esc(header_path)   << "\",\n";
        out << "  \"fp_radius\": "  << fp_radius  << ",\n";
        out << "  \"fp_nbits\": "   << fp_nbits   << ",\n";
        out << "  \"nlist\": "      << nlist      << ",\n";
        out << "  \"pq_m\": "       << pq_m       << ",\n";
        out << "  \"pq_nbits\": "   << pq_nbits   << ",\n";
        out << "  \"shards\": [\n";
        for (size_t i = 0; i < shards.size(); i++) {
            const auto& s = shards[i];
            out << "    {"
                << "\"id\": " << s.id << ", "
                << "\"mol_start\": " << s.mol_start << ", "
                << "\"mol_end\": "   << s.mol_end   << ", "
                << "\"faiss_path\": \"" << esc(s.faiss_path) << "\""
                << "}";
            if (i + 1 < shards.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n}\n";
    }

    // ── JSON read ─────────────────────────────────────────────────────────────
    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("ShardConfig: cannot read " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string json = ss.str();

        total_molecules = get_uint(json, "total_molecules");
        cxsmiles_path   = get_str (json, "cxsmiles_path");
        offsets_path    = get_str (json, "offsets_path");
        header_path     = get_str (json, "header_path");
        fp_radius       = (int)get_uint(json, "fp_radius",  3);
        fp_nbits        = (int)get_uint(json, "fp_nbits",   1024);
        nlist           = (int)get_uint(json, "nlist",      4096);
        pq_m            = (int)get_uint(json, "pq_m",       16);
        pq_nbits        = (int)get_uint(json, "pq_nbits",   8);

        shards.clear();
        // Find the "shards": [ ... ] array
        auto arr_start = json.find("\"shards\"");
        if (arr_start == std::string::npos) return;
        auto bracket = json.find('[', arr_start);
        if (bracket == std::string::npos) return;

        // Walk character-by-character, extract each { ... } object at depth 1
        int depth = 0;
        size_t obj_start = std::string::npos;
        for (size_t pos = bracket; pos < json.size(); ++pos) {
            char c = json[pos];
            if (c == '[' || c == '{') {
                if (c == '{' && depth == 1) obj_start = pos;
                ++depth;
            } else if (c == ']' || c == '}') {
                --depth;
                if (c == '}' && depth == 1 && obj_start != std::string::npos) {
                    std::string obj = json.substr(obj_start, pos - obj_start + 1);
                    ShardEntry e;
                    e.id        = (int)get_uint(obj, "id",        0);
                    e.mol_start = get_uint(obj, "mol_start", 0);
                    e.mol_end   = get_uint(obj, "mol_end",   0);
                    e.faiss_path= get_str (obj, "faiss_path");
                    shards.push_back(e);
                    obj_start = std::string::npos;
                }
                if (depth == 0) break;
            }
        }
    }

private:
    static std::string get_str(const std::string& json, const std::string& key) {
        std::string needle = "\"" + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) return {};
        pos = json.find(':', pos + needle.size());
        if (pos == std::string::npos) return {};
        pos = json.find('"', pos);
        if (pos == std::string::npos) return {};
        ++pos;
        std::string r;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                char next = json[pos + 1];
                if (next == '"')       { r += '"';  pos += 2; }
                else if (next == '\\') { r += '\\'; pos += 2; }
                else { r += json[pos++]; }
            } else {
                r += json[pos++];
            }
        }
        return r;
    }

    static uint64_t get_uint(const std::string& json, const std::string& key,
                             uint64_t def = 0) {
        std::string needle = "\"" + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) return def;
        pos = json.find(':', pos + needle.size());
        if (pos == std::string::npos) return def;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
            ++pos;
        if (pos >= json.size() || json[pos] < '0' || json[pos] > '9') return def;
        uint64_t v = 0;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9')
            v = v * 10 + (json[pos++] - '0');
        return v;
    }
};

} // namespace bscs
