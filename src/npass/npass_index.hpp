#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/logger.hpp"
#include "../common/tsv_utils.hpp"
#include "npass_types.hpp"

namespace bscs {
namespace fs = std::filesystem;

// ─── helpers ─────────────────────────────────────────────────────────────────

static NpassRow row_to_map(const std::vector<std::string>& header,
                           const std::vector<std::string_view>& fields) {
    NpassRow m;
    for (size_t i = 0; i < header.size() && i < fields.size(); ++i)
        m[header[i]] = std::string(fields[i]);
    return m;
}

// Load a TSV file (with header) into a map keyed by a given key column.
// Multi-value: key → vector of rows.
static void load_tsv_multi(
        const std::string& path,
        const std::string& key_col,
        std::unordered_map<std::string, std::vector<NpassRow>>& out) {

    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_WARN("load_tsv_multi: cannot open %s (skipping)", path.c_str());
        return;
    }
    std::string line;
    if (!std::getline(f, line)) return;
    auto header = parse_header(trim_right(line));
    int ki = col_index(header, key_col);
    if (ki < 0) {
        LOG_WARN("load_tsv_multi: key column '%s' not in %s", key_col.c_str(), path.c_str());
        return;
    }
    std::vector<std::string_view> fields;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        split_tsv(std::string_view(trim_right(line)), fields);
        if ((int)fields.size() <= ki) continue;
        std::string key(fields[ki]);
        out[key].push_back(row_to_map(header, fields));
    }
}

// Single-value map: key → one row.
static void load_tsv_single(
        const std::string& path,
        const std::string& key_col,
        std::unordered_map<std::string, NpassRow>& out) {

    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_WARN("load_tsv_single: cannot open %s (skipping)", path.c_str());
        return;
    }
    std::string line;
    if (!std::getline(f, line)) return;
    auto header = parse_header(trim_right(line));
    int ki = col_index(header, key_col);
    if (ki < 0) return;
    std::vector<std::string_view> fields;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        split_tsv(std::string_view(trim_right(line)), fields);
        if ((int)fields.size() <= ki) continue;
        std::string key(fields[ki]);
        if (!out.count(key))
            out[key] = row_to_map(header, fields);
    }
}

// ─── NpassFpStore ────────────────────────────────────────────────────────────
// Pre-computed Morgan fingerprints written by prepare_npass (Phase 3).
// Enables ~20× faster brute-force Tanimoto search: no SMILES parsing per query.
struct NpassFpStore {
    int nbits   = 0;
    int radius  = 0;
    int n_words = 0;                  // nbits / 64
    std::vector<std::string> npids;   // compound order matches words
    std::vector<uint64_t>    words;   // flat: npids.size() × n_words

    bool loaded() const { return nbits > 0 && !npids.empty(); }

    void load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open " + path);

        uint32_t magic = 0, version = 0, n = 0, nb = 0, rad = 0;
        if (std::fread(&magic, 4, 1, f) != 1 || magic != 0x4E504650u) {
            std::fclose(f);
            throw std::runtime_error("npass_fps.bin: bad magic");
        }
        std::fread(&version, 4, 1, f);
        std::fread(&n,       4, 1, f);
        std::fread(&nb,      4, 1, f);
        std::fread(&rad,     4, 1, f);

        nbits   = (int)nb;
        radius  = (int)rad;
        n_words = nbits / 64;
        int bytes_per_fp = nbits / 8;

        npids.reserve(n);
        words.assign((size_t)n * n_words, 0);

        std::vector<uint8_t> buf(bytes_per_fp);
        for (uint32_t i = 0; i < n; i++) {
            uint16_t npid_len = 0;
            std::fread(&npid_len, 2, 1, f);
            std::string npid(npid_len, '\0');
            std::fread(npid.data(), 1, npid_len, f);
            std::fread(buf.data(), 1, bytes_per_fp, f);

            uint64_t* dst = words.data() + (size_t)i * n_words;
            for (int b = 0; b < nbits; b++)
                if (buf[b/8] & (uint8_t)(1u << (b % 8)))
                    dst[b/64] |= (1ULL << (b % 64));
            npids.push_back(std::move(npid));
        }
        std::fclose(f);
    }

    static int popcount(const uint64_t* w, int nw) {
        int c = 0;
        for (int i = 0; i < nw; i++) c += __builtin_popcountll(w[i]);
        return c;
    }

    double tanimoto(const uint64_t* qw, int q_pop, size_t ci) const {
        const uint64_t* cw = words.data() + ci * n_words;
        int both = 0;
        for (int i = 0; i < n_words; i++) both += __builtin_popcountll(qw[i] & cw[i]);
        int c_pop = popcount(cw, n_words);
        int uni   = q_pop + c_pop - both;
        return uni > 0 ? (double)both / uni : 0.0;
    }
};

// ─── NpassIndex ──────────────────────────────────────────────────────────────

// In-RAM relational index for NPASS.  NPASS is small enough (~30-80K compounds)
// to fit entirely in memory.
class NpassIndex {
public:
    NpassIndex() = default;

    // Build from NPASS directory.  Call this in prepare_npass and save(),
    // or call load_from_dir() directly in search_annotate.
    void load_from_dir(const std::string& npass_dir) {
        auto p = [&](const char* name) {
            return (fs::path(npass_dir) / name).string();
        };

        LOG_INFO("Loading NPASS structure...");
        load_structure(p("NPASS3.0_naturalproducts_structure.txt"));

        LOG_INFO("Loading NPASS generalinfo...");
        load_tsv_single(p("NPASS3.0_naturalproducts_generalinfo.txt"),
                        "np_id", generalinfo_);

        LOG_INFO("Loading NPASS activities...");
        load_tsv_multi(p("NPASS3.0_activities.txt"),    "np_id", activities_by_npid_);

        LOG_INFO("Loading NPASS toxicity...");
        load_tsv_multi(p("NPASS3.0_toxicity.txt"),      "np_id", toxicity_by_npid_);

        LOG_INFO("Loading NPASS species_pair...");
        load_tsv_multi(p("NPASS3.0_naturalproducts_species_pair.txt"),
                       "np_id", species_pair_by_npid_);

        LOG_INFO("Loading NPASS targets...");
        load_tsv_single(p("NPASS3.0_target.txt"),       "target_id", targets_);

        LOG_INFO("Loading NPASS species_info...");
        load_tsv_single(p("NPASS3.0_species_info.txt"), "org_id",    species_info_);

        LOG_INFO("Loading NPASS coculture...");
        load_tsv_multi(p("NPASS3.0_Coculture.tsv"),     "np_id", coculture_);

        LOG_INFO("Loading NPASS elicitation...");
        load_tsv_multi(p("NPASS3.0_Elicitation.tsv"),   "np_id", elicitation_);

        LOG_INFO("Loading NPASS engineer...");
        load_tsv_multi(p("NPASS3.0_Engineer.tsv"),      "np_id", engineer_);

        LOG_INFO("Loading NPASS symbiont...");
        load_tsv_multi(p("NPASS3.0_Symbiont.tsv"),      "np_id", symbiont_);

        LOG_INFO("NPASS loaded: %zu compounds, %zu activities, %zu toxicity records",
                 by_inchikey_.size(), total_activities(), total_toxicity());
    }

    // Lookup by InChIKey (primary) or canonical SMILES (fallback).
    // Returns empty vector if no match.
    std::vector<NpassCompound> lookup(const std::string& inchikey,
                                      const std::string& canon_smiles = {}) const {
        std::vector<NpassCompound> result;

        std::vector<std::string> np_ids;
        if (!inchikey.empty()) {
            auto it = by_inchikey_.find(inchikey);
            if (it != by_inchikey_.end()) np_ids = it->second;
        }
        if (np_ids.empty() && !canon_smiles.empty()) {
            auto it = by_smiles_.find(canon_smiles);
            if (it != by_smiles_.end()) np_ids = it->second;
        }

        for (const std::string& np_id : np_ids)
            result.push_back(build_record(np_id));

        return result;
    }

    size_t compound_count() const { return by_inchikey_.size(); }

    // Returns {np_id, smiles} for all compounds that have SMILES — used for
    // brute-force similarity search across the full NPASS corpus.
    std::vector<std::pair<std::string,std::string>> all_smiles() const {
        std::vector<std::pair<std::string,std::string>> out;
        out.reserve(structure_.size());
        for (const auto& [np_id, arr] : structure_)
            if (!arr[3].empty()) out.emplace_back(np_id, arr[3]);
        return out;
    }

    NpassCompound lookup_by_npid(const std::string& np_id) const {
        return build_record(np_id);
    }

private:
    // InChIKey  →  np_id(s)
    std::unordered_map<std::string, std::vector<std::string>> by_inchikey_;
    // canonical SMILES → np_id(s)  (fallback)
    std::unordered_map<std::string, std::vector<std::string>> by_smiles_;

    // np_id → structure fields
    std::unordered_map<std::string, std::array<std::string,4>> structure_;
    // np_id → generalinfo row
    std::unordered_map<std::string, NpassRow> generalinfo_;
    // np_id → multi activity rows
    std::unordered_map<std::string, std::vector<NpassRow>> activities_by_npid_;
    std::unordered_map<std::string, std::vector<NpassRow>> toxicity_by_npid_;
    std::unordered_map<std::string, std::vector<NpassRow>> species_pair_by_npid_;
    std::unordered_map<std::string, std::vector<NpassRow>> coculture_;
    std::unordered_map<std::string, std::vector<NpassRow>> elicitation_;
    std::unordered_map<std::string, std::vector<NpassRow>> engineer_;
    std::unordered_map<std::string, std::vector<NpassRow>> symbiont_;
    // target_id → target row
    std::unordered_map<std::string, NpassRow> targets_;
    // org_id → species row
    std::unordered_map<std::string, NpassRow> species_info_;

    void load_structure(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_WARN("Cannot open structure file: %s", path.c_str());
            return;
        }
        std::string line;
        if (!std::getline(f, line)) return;
        auto hdr = parse_header(trim_right(line));
        int i_npid = col_index(hdr, "np_id");
        int i_inchi = col_index(hdr, "InChI");
        int i_ik    = col_index(hdr, "InChIKey");
        int i_smi   = col_index(hdr, "SMILES");
        std::vector<std::string_view> flds;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            split_tsv(std::string_view(trim_right(line)), flds);
            auto get = [&](int idx) -> std::string {
                if (idx < 0 || idx >= (int)flds.size()) return {};
                return std::string(flds[idx]);
            };
            std::string np_id = get(i_npid);
            if (np_id.empty()) continue;
            std::string ik  = get(i_ik);
            std::string smi = get(i_smi);
            structure_[np_id] = {np_id, get(i_inchi), ik, smi};
            if (!ik.empty())  by_inchikey_[ik].push_back(np_id);
            if (!smi.empty()) by_smiles_[smi].push_back(np_id);
        }
    }

    NpassCompound build_record(const std::string& np_id) const {
        NpassCompound c;
        c.np_id = np_id;

        auto sit = structure_.find(np_id);
        if (sit != structure_.end()) {
            c.inchi       = sit->second[1];
            c.inchikey    = sit->second[2];
            c.smiles_npass= sit->second[3];
        }

        auto git = generalinfo_.find(np_id);
        if (git != generalinfo_.end()) c.generalinfo = git->second;

        // Activities joined with target table
        auto ait = activities_by_npid_.find(np_id);
        if (ait != activities_by_npid_.end()) {
            for (const NpassRow& act : ait->second) {
                NpassCompound::ActivityRecord ar;
                ar.activity = act;
                auto tid_it = act.find("target_id");
                if (tid_it != act.end()) {
                    auto tit = targets_.find(tid_it->second);
                    if (tit != targets_.end()) ar.target = tit->second;
                }
                c.activities.push_back(std::move(ar));
            }
        }

        // Toxicity joined with target table
        auto tit = toxicity_by_npid_.find(np_id);
        if (tit != toxicity_by_npid_.end()) {
            for (const NpassRow& tox : tit->second) {
                NpassCompound::ToxicityRecord tr;
                tr.toxicity = tox;
                auto tid_it = tox.find("target_id");
                if (tid_it != tox.end()) {
                    auto ttit = targets_.find(tid_it->second);
                    if (ttit != targets_.end()) tr.target = ttit->second;
                }
                c.toxicities.push_back(std::move(tr));
            }
        }

        // Species pairs joined with species_info
        auto spin = species_pair_by_npid_.find(np_id);
        if (spin != species_pair_by_npid_.end()) {
            for (const NpassRow& sp : spin->second) {
                NpassCompound::SpeciesPairRecord spr;
                spr.species_pair = sp;
                auto oid_it = sp.find("org_id");
                if (oid_it != sp.end()) {
                    auto siit = species_info_.find(oid_it->second);
                    if (siit != species_info_.end()) spr.species_info = siit->second;
                }
                c.species_pairs.push_back(std::move(spr));
            }
        }

        auto copy_multi = [&](
                const std::unordered_map<std::string, std::vector<NpassRow>>& tbl,
                std::vector<NpassRow>& dst) {
            auto it = tbl.find(np_id);
            if (it != tbl.end()) dst = it->second;
        };
        copy_multi(coculture_,   c.coculture_records);
        copy_multi(elicitation_, c.elicitation_records);
        copy_multi(engineer_,    c.engineer_records);
        copy_multi(symbiont_,    c.symbiont_records);

        return c;
    }

    size_t total_activities() const {
        size_t n = 0;
        for (auto& kv : activities_by_npid_) n += kv.second.size();
        return n;
    }
    size_t total_toxicity() const {
        size_t n = 0;
        for (auto& kv : toxicity_by_npid_) n += kv.second.size();
        return n;
    }
};

} // namespace bscs
