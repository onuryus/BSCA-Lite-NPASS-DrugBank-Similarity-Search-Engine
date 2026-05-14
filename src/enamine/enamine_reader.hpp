#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

#include "../common/mmap_file.hpp"
#include "../common/tsv_utils.hpp"
#include "offset_store.hpp"

namespace bscs {

// Full parsed Enamine record preserving all original columns.
struct EnamineRecord {
    uint64_t mol_id = 0;                   // 0-based sequential ID = FAISS ID
    std::vector<std::string> fields;        // all original TSV fields
    const std::vector<std::string>* header; // pointer to shared header (not owned)

    std::string_view get(const std::string& col_name) const {
        int idx = col_index(*header, col_name);
        if (idx < 0 || idx >= (int)fields.size()) return {};
        return fields[idx];
    }

    std::string smiles() const {
        return fields.empty() ? std::string{} : strip_cxsmiles(fields[0]);
    }

    std::string inchikey() const {
        // Column is "InChiKey" in Enamine (mixed case)
        for (const char* name : {"InChiKey","InChIKey","inchikey"}) {
            int idx = col_index(*header, name);
            if (idx >= 0 && idx < (int)fields.size())
                return std::string(fields[idx]);
        }
        return {};
    }
};

// Retrieves Enamine records from mmap'd file using precomputed offsets.
class EnamineReader {
public:
    EnamineReader() = default;

    void open(const std::string& enamine_path,
              const std::string& offsets_path,
              const std::string& header_path) {
        file_.open(enamine_path);
        file_.hint_random();
        offsets_.open(offsets_path);
        load_header(header_path);
    }

    // Fetch a single record by molecule ID.
    EnamineRecord fetch(uint64_t mol_id) const {
        uint64_t off = offsets_[mol_id];
        std::string line = file_.read_line(off);
        return parse_line(mol_id, line);
    }

    uint64_t count() const { return offsets_.count(); }
    const std::vector<std::string>& header() const { return header_; }

    // Column index by name (or -1).
    int col_idx(const std::string& name) const {
        return col_index(header_, name);
    }

private:
    MmapFile    file_;
    OffsetStore offsets_;
    std::vector<std::string> header_;

    void load_header(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) throw std::runtime_error("EnamineReader: cannot open header " + path);
        char buf[4096];
        if (std::fgets(buf, sizeof(buf), f)) {
            std::string line(buf);
            // strip trailing newline
            while (!line.empty() && (line.back()=='\n'||line.back()=='\r'))
                line.pop_back();
            header_ = parse_header(line);
        }
        std::fclose(f);
    }

    EnamineRecord parse_line(uint64_t mol_id, const std::string& line) const {
        EnamineRecord rec;
        rec.mol_id = mol_id;
        rec.header = &header_;
        auto sv = split_tsv(std::string_view(line));
        rec.fields.reserve(sv.size());
        for (auto& f : sv) rec.fields.emplace_back(f);
        // Pad to header width so get() never goes out of range
        while (rec.fields.size() < header_.size()) rec.fields.emplace_back();
        return rec;
    }
};

} // namespace bscs
