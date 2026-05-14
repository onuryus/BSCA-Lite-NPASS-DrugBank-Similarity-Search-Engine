// prepare_npass: validates NPASS tables and writes binary lookup indexes.
// Also pre-computes Morgan fingerprints and saves npass_fps.bin for fast search.
//
// Usage:
//   prepare_npass --npass-dir DIR --outdir DIR

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>
#include <sys/stat.h>

#include "common/fp_utils.hpp"
#include "common/logger.hpp"
#include "common/progress.hpp"
#include "common/tsv_utils.hpp"
#include "npass/npass_index.hpp"

namespace fs = std::filesystem;
using namespace bscs;

// ── binary serialization helpers ─────────────────────────────────────────────
// Per-table binary format (multi-value):
//   uint64_t  num_keys
//   for each key:
//     uint16_t key_len,  char[key_len]
//     uint32_t num_rows
//     for each row:
//       uint32_t num_cols
//       for each col:
//         uint16_t name_len, char[name_len]
//         uint16_t val_len,  char[val_len]
//
// Single-value tables: same but num_rows is always 1.

static void write_u16(FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }
static void write_u32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }

static void write_string(FILE* f, const std::string& s) {
    uint16_t len = (uint16_t)std::min(s.size(), (size_t)65535u);
    write_u16(f, len);
    if (len) std::fwrite(s.data(), 1, len, f);
}

static void write_row(FILE* f, const NpassRow& row) {
    write_u32(f, (uint32_t)row.size());
    for (const auto& [k, v] : row) {
        write_string(f, k);
        write_string(f, v);
    }
}

static void write_multi_table(
        FILE* f,
        const std::unordered_map<std::string, std::vector<NpassRow>>& tbl) {
    write_u64(f, (uint64_t)tbl.size());
    for (const auto& [key, rows] : tbl) {
        write_string(f, key);
        write_u32(f, (uint32_t)rows.size());
        for (const auto& row : rows) write_row(f, row);
    }
}

static void write_single_table(
        FILE* f,
        const std::unordered_map<std::string, NpassRow>& tbl) {
    write_u64(f, (uint64_t)tbl.size());
    for (const auto& [key, row] : tbl) {
        write_string(f, key);
        write_row(f, row);
    }
}

static void write_inchikey_index(
        FILE* f,
        const std::unordered_map<std::string, std::vector<std::string>>& by_ik) {
    std::vector<std::pair<std::string, std::vector<std::string>>> entries(
        by_ik.begin(), by_ik.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    write_u64(f, (uint64_t)entries.size());
    for (const auto& [ik, npids] : entries) {
        char ik27[27] = {};
        std::memcpy(ik27, ik.data(), std::min(ik.size(), (size_t)27));
        std::fwrite(ik27, 1, 27, f);
        write_u32(f, (uint32_t)npids.size());
        for (const auto& npid : npids) write_string(f, npid);
    }
}

static void write_smiles_index(
        FILE* f,
        const std::unordered_map<std::string, std::vector<std::string>>& by_smi) {
    std::vector<std::pair<std::string, std::vector<std::string>>> entries(
        by_smi.begin(), by_smi.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    write_u64(f, (uint64_t)entries.size());
    for (const auto& [smi, npids] : entries) {
        write_string(f, smi);
        write_u32(f, (uint32_t)npids.size());
        for (const auto& npid : npids) write_string(f, npid);
    }
}

static void write_structure_table(
        FILE* f,
        const std::unordered_map<std::string, std::array<std::string,4>>& sm) {
    write_u64(f, (uint64_t)sm.size());
    for (const auto& [npid, arr] : sm) {
        write_string(f, npid);
        for (const auto& s : arr) write_string(f, s);
    }
}

// ── timed table loader wrappers ───────────────────────────────────────────────
static void timed_load_multi(
        const std::string& path, const std::string& key_col,
        std::unordered_map<std::string, std::vector<NpassRow>>& out,
        const char* label) {

    std::fprintf(stderr, "  Loading %-45s ", label);
    std::fflush(stderr);
    auto t0 = std::chrono::steady_clock::now();
    load_tsv_multi(path, key_col, out);
    double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    size_t total_rows = 0;
    for (const auto& [k, v] : out) total_rows += v.size();
    std::fprintf(stderr, "-> %6zu keys  %8zu rows  [%.2fs]\n",
                 out.size(), total_rows, dt);
    std::fflush(stderr);
}

static void timed_load_single(
        const std::string& path, const std::string& key_col,
        std::unordered_map<std::string, NpassRow>& out,
        const char* label) {

    std::fprintf(stderr, "  Loading %-45s ", label);
    std::fflush(stderr);
    auto t0 = std::chrono::steady_clock::now();
    load_tsv_single(path, key_col, out);
    double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "-> %6zu entries  [%.2fs]\n", out.size(), dt);
    std::fflush(stderr);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string npass_dir;
    std::string outdir;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--npass-dir" && i+1 < argc) npass_dir = argv[++i];
        else if (a == "--outdir"    && i+1 < argc) outdir    = argv[++i];
    }

    if (npass_dir.empty() || outdir.empty()) {
        std::cerr << "Usage: prepare_npass --npass-dir DIR --outdir DIR\n";
        return 1;
    }

    fs::create_directories(outdir);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_npass\n");
    std::fprintf(stderr, "  NPASS dir: %s\n", npass_dir.c_str());
    std::fprintf(stderr, "  Out dir  : %s\n", outdir.c_str());
    std::fprintf(stderr, "=================================================================\n\n");

    auto t_total = std::chrono::steady_clock::now();

    auto p = [&](const char* name) {
        return (fs::path(npass_dir) / name).string();
    };

    using MultiMap  = std::unordered_map<std::string, std::vector<NpassRow>>;
    using SingleMap = std::unordered_map<std::string, NpassRow>;
    using StructMap = std::unordered_map<std::string, std::array<std::string,4>>;
    using IKMap     = std::unordered_map<std::string, std::vector<std::string>>;

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: load all NPASS tables
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 1: loading NPASS tables\n");

    StructMap struct_map;
    IKMap     by_inchikey, by_smiles;

    // Structure file (special: builds InChIKey and SMILES lookup maps)
    {
        std::fprintf(stderr, "  Loading %-45s ", "NPASS3.0_naturalproducts_structure.txt");
        std::fflush(stderr);
        auto t0 = std::chrono::steady_clock::now();

        std::ifstream f(p("NPASS3.0_naturalproducts_structure.txt"));
        if (!f.is_open()) { LOG_ERR("Cannot open structure file"); return 1; }
        std::string line;
        std::getline(f, line);
        auto hdr   = parse_header(trim_right(line));
        int i_npid = col_index(hdr, "np_id");
        int i_inchi= col_index(hdr, "InChI");
        int i_ik   = col_index(hdr, "InChIKey");
        int i_smi  = col_index(hdr, "SMILES");
        std::vector<std::string_view> flds;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            split_tsv(std::string_view(trim_right(line)), flds);
            auto get = [&](int idx) -> std::string {
                if (idx < 0 || idx >= (int)flds.size()) return {};
                return std::string(flds[idx]);
            };
            std::string npid = get(i_npid);
            if (npid.empty()) continue;
            std::string ik  = get(i_ik);
            std::string smi = get(i_smi);
            struct_map[npid] = {npid, get(i_inchi), ik, smi};
            if (!ik.empty())  by_inchikey[ik].push_back(npid);
            if (!smi.empty()) by_smiles[smi].push_back(npid);
        }

        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "-> %6zu compounds  [%.2fs]\n", struct_map.size(), dt);
        std::fflush(stderr);
    }

    SingleMap generalinfo;
    timed_load_single(p("NPASS3.0_naturalproducts_generalinfo.txt"), "np_id",
                      generalinfo, "NPASS3.0_naturalproducts_generalinfo.txt");

    MultiMap activities, toxicity, species_pair, coculture, elicitation, engineer, symbiont;
    timed_load_multi(p("NPASS3.0_activities.txt"),    "np_id", activities,
                     "NPASS3.0_activities.txt");
    timed_load_multi(p("NPASS3.0_toxicity.txt"),      "np_id", toxicity,
                     "NPASS3.0_toxicity.txt");
    timed_load_multi(p("NPASS3.0_naturalproducts_species_pair.txt"), "np_id", species_pair,
                     "NPASS3.0_naturalproducts_species_pair.txt");
    timed_load_multi(p("NPASS3.0_Coculture.tsv"),     "np_id", coculture,
                     "NPASS3.0_Coculture.tsv");
    timed_load_multi(p("NPASS3.0_Elicitation.tsv"),   "np_id", elicitation,
                     "NPASS3.0_Elicitation.tsv");
    timed_load_multi(p("NPASS3.0_Engineer.tsv"),       "np_id", engineer,
                     "NPASS3.0_Engineer.tsv");
    timed_load_multi(p("NPASS3.0_Symbiont.tsv"),       "np_id", symbiont,
                     "NPASS3.0_Symbiont.tsv");

    SingleMap targets, species_info;
    timed_load_single(p("NPASS3.0_target.txt"),       "target_id", targets,
                      "NPASS3.0_target.txt");
    timed_load_single(p("NPASS3.0_species_info.txt"), "org_id",    species_info,
                      "NPASS3.0_species_info.txt");

    // Summary
    {
        size_t act_rows = 0; for (auto& [k,v]: activities)   act_rows  += v.size();
        size_t tox_rows = 0; for (auto& [k,v]: toxicity)     tox_rows  += v.size();
        size_t sp_rows  = 0; for (auto& [k,v]: species_pair) sp_rows   += v.size();
        std::fprintf(stderr, "\n  Phase 1 summary:\n");
        std::fprintf(stderr, "    Compounds    : %s\n", fmt_human(struct_map.size()).c_str());
        std::fprintf(stderr, "    InChIKey keys: %s\n", fmt_human(by_inchikey.size()).c_str());
        std::fprintf(stderr, "    SMILES keys  : %s\n", fmt_human(by_smiles.size()).c_str());
        std::fprintf(stderr, "    Activities   : %s rows\n", fmt_human(act_rows).c_str());
        std::fprintf(stderr, "    Toxicity     : %s rows\n", fmt_human(tox_rows).c_str());
        std::fprintf(stderr, "    Species pairs: %s rows\n", fmt_human(sp_rows).c_str());
        std::fprintf(stderr, "    Targets      : %s\n", fmt_human(targets.size()).c_str());
        std::fprintf(stderr, "    Species info : %s\n\n", fmt_human(species_info.size()).c_str());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: write all binary index files (13 files + 2 lookup indexes)
    // ─────────────────────────────────────────────────────────────────────────
    struct WriteTask {
        const char* filename;
        const char* description;
    };
    const WriteTask tasks[] = {
        {"inchikey_index.bin", "InChIKey → np_id sorted lookup"},
        {"smiles_index.bin",   "canonical SMILES → np_id sorted lookup"},
        {"structure.bin",      "np_id → {InChI, InChIKey, SMILES}"},
        {"generalinfo.bin",    "np_id → general info record"},
        {"targets.bin",        "target_id → target record"},
        {"species_info.bin",   "org_id → species info record"},
        {"activities.bin",     "np_id → [activity rows]"},
        {"toxicity.bin",       "np_id → [toxicity rows]"},
        {"species_pair.bin",   "np_id → [species pair rows]"},
        {"coculture.bin",      "np_id → [coculture rows]"},
        {"elicitation.bin",    "np_id → [elicitation rows]"},
        {"engineer.bin",       "np_id → [engineer rows]"},
        {"symbiont.bin",       "np_id → [symbiont rows]"},
    };
    const int N_TASKS = (int)(sizeof(tasks) / sizeof(tasks[0]));

    Progress p2("Phase 2: writing binary index files", N_TASKS, "files");
    int task_done = 0;

    auto write_bin = [&](const char* filename, auto&& fn) {
        std::string path = (fs::path(outdir) / filename).string();
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { LOG_ERR("Cannot write %s", path.c_str()); return; }
        fn(f);
        std::fclose(f);
        // Get file size for display
        struct stat st{};
        ::stat(path.c_str(), &st);
        // Print completion inline (after progress line)
        std::fprintf(stderr, "\r    [OK] %-35s  %s\n",
                     filename, fmt_bytes((uint64_t)st.st_size).c_str());
        std::fflush(stderr);
        p2.update(++task_done);
    };

    std::fprintf(stderr, "\n[PHASE] Phase 2: writing binary index files  (%d files)\n",
                 N_TASKS);

    write_bin("inchikey_index.bin", [&](FILE* f) { write_inchikey_index(f, by_inchikey); });
    write_bin("smiles_index.bin",   [&](FILE* f) { write_smiles_index(f, by_smiles); });
    write_bin("structure.bin",      [&](FILE* f) { write_structure_table(f, struct_map); });
    write_bin("generalinfo.bin",    [&](FILE* f) { write_single_table(f, generalinfo); });
    write_bin("targets.bin",        [&](FILE* f) { write_single_table(f, targets); });
    write_bin("species_info.bin",   [&](FILE* f) { write_single_table(f, species_info); });
    write_bin("activities.bin",     [&](FILE* f) { write_multi_table(f, activities); });
    write_bin("toxicity.bin",       [&](FILE* f) { write_multi_table(f, toxicity); });
    write_bin("species_pair.bin",   [&](FILE* f) { write_multi_table(f, species_pair); });
    write_bin("coculture.bin",      [&](FILE* f) { write_multi_table(f, coculture); });
    write_bin("elicitation.bin",    [&](FILE* f) { write_multi_table(f, elicitation); });
    write_bin("engineer.bin",       [&](FILE* f) { write_multi_table(f, engineer); });
    write_bin("symbiont.bin",       [&](FILE* f) { write_multi_table(f, symbiont); });

    p2.finish(task_done);

    // Manifest
    {
        std::ofstream mf((fs::path(outdir) / "npass_index.manifest").string());
        mf << "version=1\n";
        mf << "compounds=" << struct_map.size() << "\n";
        mf << "inchikey_entries=" << by_inchikey.size() << "\n";
        mf << "smiles_entries=" << by_smiles.size() << "\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 3: pre-compute Morgan fingerprints → npass_fps.bin
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "\n[PHASE] Phase 3: pre-computing Morgan fingerprints (radius=3, nbits=1024)\n");
    {
        const int nbits  = 1024;
        const int radius = 3;
        const int bytes_per_fp = nbits / 8;
        FpConfig fp_cfg{radius, nbits};

        // Collect all {npid, smiles} pairs with valid SMILES
        struct SmiEntry { std::string npid; std::string smiles; };
        std::vector<SmiEntry> entries;
        entries.reserve(struct_map.size());
        for (const auto& [npid, arr] : struct_map)
            if (!arr[3].empty())
                entries.push_back({npid, arr[3]});

        int n = (int)entries.size();
        std::vector<uint8_t> all_fps((size_t)n * bytes_per_fp, 0);
        std::vector<bool>    ok(n, false);

        auto t_fp = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            auto mol = smiles_to_mol(entries[i].smiles);
            if (!mol) continue;
            auto fp = mol_to_fp_bv(*mol, fp_cfg);
            if (!fp) continue;
            uint8_t* dst = all_fps.data() + (size_t)i * bytes_per_fp;
            for (int b = 0; b < nbits; b++)
                if (fp->getBit(b)) dst[b/8] |= (uint8_t)(1u << (b % 8));
            ok[i] = true;
        }

        int valid = 0;
        for (bool b : ok) if (b) valid++;
        double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_fp).count();
        std::fprintf(stderr, "  [OK]  %d / %d compounds fingerprinted  [%.2fs]\n",
                     valid, n, dt);

        // Write npass_fps.bin
        std::string fps_path = (fs::path(outdir) / "npass_fps.bin").string();
        FILE* fout = std::fopen(fps_path.c_str(), "wb");
        if (!fout) {
            LOG_WARN("Cannot write %s", fps_path.c_str());
        } else {
            uint32_t magic   = 0x4E504650u; // 'NPFP'
            uint32_t version = 1;
            uint32_t n32     = (uint32_t)valid;
            uint32_t nbits32 = (uint32_t)nbits;
            uint32_t rad32   = (uint32_t)radius;
            std::fwrite(&magic,   4, 1, fout);
            std::fwrite(&version, 4, 1, fout);
            std::fwrite(&n32,     4, 1, fout);
            std::fwrite(&nbits32, 4, 1, fout);
            std::fwrite(&rad32,   4, 1, fout);
            for (int i = 0; i < n; i++) {
                if (!ok[i]) continue;
                uint16_t npid_len = (uint16_t)entries[i].npid.size();
                std::fwrite(&npid_len,           2, 1,        fout);
                std::fwrite(entries[i].npid.data(), 1, npid_len, fout);
                std::fwrite(all_fps.data() + (size_t)i * bytes_per_fp,
                            1, bytes_per_fp, fout);
            }
            std::fclose(fout);
            struct stat st{}; ::stat(fps_path.c_str(), &st);
            std::fprintf(stderr, "  [OK]  Saved: %s  (%s)\n",
                         fps_path.c_str(), fmt_bytes((uint64_t)st.st_size).c_str());
        }
    }

    double total_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_total).count();

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_npass complete.\n");
    std::fprintf(stderr, "  Total time : %s\n", fmt_elapsed(total_sec).c_str());
    std::fprintf(stderr, "  Compounds  : %s\n", fmt_human(struct_map.size()).c_str());
    std::fprintf(stderr, "  Output dir : %s\n", outdir.c_str());
    std::fprintf(stderr, "=================================================================\n\n");

    return 0;
}
