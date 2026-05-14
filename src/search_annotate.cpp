// search_annotate: FAISS search + exact rerank + Enamine / NPASS / DrugBank annotation.
// Supports persistent session mode: databases loaded once, multiple queries via interactive loop.
//
// Usage:
//   search_annotate
//     --faiss  index/enamine/faiss.index
//     --enamine 2026.01_Enamine_REAL_DB_136M.cxsmiles
//     --enamine-offsets index/enamine/enamine.offsets.bin
//     --enamine-header  index/enamine/enamine.header.tsv   (optional)
//     [--query "SMILES"]            (first query; if omitted, asked interactively)
//     [--npass-index index/npass]
//     [--drugbank-bin index/drugbank/drugbank.bin]
//     [--databases enamine,npass,drugbank]
//     [--out-mode combined|per-db|both]
//     [--metric tanimoto|dice|tversky|cosine|kulczynski]
//     [--k 5000]     [--top 50]     [--nprobe 64]
//     [--radius 3]   [--nbits 4096]
//     [--out results.tsv]
//     [--profile]    (print per-query timing summary after each search)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include "common/fp_utils.hpp"
#include "common/logger.hpp"
#include "common/progress.hpp"
#include "common/similarity.hpp"
#include "common/tsv_utils.hpp"
#include "drugbank/drugbank_index.hpp"
#include "enamine/enamine_reader.hpp"
#include "npass/npass_index.hpp"

namespace fs = std::filesystem;
using namespace bscs;

using Clock = std::chrono::steady_clock;
static double elapsed_s(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// ── CLI ───────────────────────────────────────────────────────────────────────
struct Config {
    std::string query_smiles;        // optional; if empty, asked interactively
    std::string faiss_path;
    std::string enamine_path;
    std::string offsets_path;
    std::string header_path;
    std::string npass_dir;
    std::string npass_raw_dir;
    std::string drugbank_bin_path;
    std::string databases_str;
    std::string out_mode_str;
    std::string out_path;
    std::string metric_str = "tanimoto";
    int    candidate_k = 5000;
    int    top_n       = 50;
    int    nprobe      = 64;
    FpConfig fp{3, 4096};
    bool   profile     = false;
};

static Config parse_args(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="--query"          &&i+1<argc) c.query_smiles      = argv[++i];
        else if (a=="--faiss"          &&i+1<argc) c.faiss_path        = argv[++i];
        else if (a=="--enamine"        &&i+1<argc) c.enamine_path      = argv[++i];
        else if (a=="--enamine-offsets"&&i+1<argc) c.offsets_path      = argv[++i];
        else if (a=="--enamine-header" &&i+1<argc) c.header_path       = argv[++i];
        else if (a=="--npass-index"    &&i+1<argc) c.npass_dir         = argv[++i];
        else if (a=="--npass-raw"      &&i+1<argc) c.npass_raw_dir     = argv[++i];
        else if (a=="--drugbank-bin"   &&i+1<argc) c.drugbank_bin_path = argv[++i];
        else if (a=="--databases"      &&i+1<argc) c.databases_str     = argv[++i];
        else if (a=="--out-mode"       &&i+1<argc) c.out_mode_str      = argv[++i];
        else if (a=="--out"            &&i+1<argc) c.out_path          = argv[++i];
        else if (a=="--metric"         &&i+1<argc) c.metric_str        = argv[++i];
        else if (a=="--k"              &&i+1<argc) c.candidate_k       = std::stoi(argv[++i]);
        else if (a=="--top"            &&i+1<argc) c.top_n             = std::stoi(argv[++i]);
        else if (a=="--nprobe"         &&i+1<argc) c.nprobe            = std::stoi(argv[++i]);
        else if (a=="--radius"         &&i+1<argc) c.fp.radius         = std::stoi(argv[++i]);
        else if (a=="--nbits"          &&i+1<argc) c.fp.nbits          = std::stoi(argv[++i]);
        else if (a=="--profile")                   c.profile           = true;
    }
    return c;
}

// ── GPU/CPU interactive selection ─────────────────────────────────────────────
static bool select_gpu_backend() {
    int num_gpus = 0;
    try { num_gpus = faiss::gpu::getNumDevices(); } catch (...) {}
    if (num_gpus == 0) return false;

    std::string gpu_name = "Unknown";
    size_t vram_mb = 0;
    try {
        const cudaDeviceProp& prop = faiss::gpu::getDeviceProperties(0);
        gpu_name = prop.name;
        vram_mb  = prop.totalGlobalMem / (1024ULL * 1024ULL);
    } catch (...) {}

    std::fprintf(stderr, "\n  GPU detected: %s (%zuMB). Use GPU? [Y/n]: ",
                 gpu_name.c_str(), vram_mb);
    std::fflush(stderr);

    std::string ans;
    std::getline(std::cin, ans);
    if (!ans.empty() && (ans[0]=='n' || ans[0]=='N')) {
        std::fprintf(stderr, "  → CPU mode selected\n");
        return false;
    }
    std::fprintf(stderr, "  → GPU mode selected\n");
    return true;
}

// ── Database selection ────────────────────────────────────────────────────────
static void parse_databases_flag(const std::string& s,
                                  bool& use_enamine, bool& use_npass, bool& use_drugbank) {
    use_enamine = use_npass = use_drugbank = false;
    std::string sl; for (char c : s) sl += (char)std::tolower((unsigned char)c);
    if (sl == "all") { use_enamine = use_npass = use_drugbank = true; return; }
    std::string token;
    for (char c : sl + ",") {
        if (c == ',') {
            if (token == "enamine")  use_enamine  = true;
            if (token == "npass")    use_npass    = true;
            if (token == "drugbank") use_drugbank = true;
            token.clear();
        } else {
            token += c;
        }
    }
}

static void show_database_menu(bool& use_enamine, bool& use_npass, bool& use_drugbank,
                                const Config& cfg) {
    bool has_npass    = !cfg.npass_dir.empty()        && fs::exists(cfg.npass_dir);
    bool has_drugbank = !cfg.drugbank_bin_path.empty() && fs::exists(cfg.drugbank_bin_path);

    std::fprintf(stderr, "\n  Select annotation databases:\n");
    std::fprintf(stderr, "  [1] Enamine only\n");
    if (has_npass)    std::fprintf(stderr, "  [2] NPASS only\n");
    if (has_drugbank) std::fprintf(stderr, "  [3] DrugBank only\n");
    if (has_npass)    std::fprintf(stderr, "  [4] Enamine + NPASS\n");
    if (has_drugbank) std::fprintf(stderr, "  [5] Enamine + DrugBank\n");
    if (has_npass && has_drugbank) {
        std::fprintf(stderr, "  [6] NPASS + DrugBank\n");
        std::fprintf(stderr, "  [7] All (Enamine + NPASS + DrugBank)\n");
    }
    std::fprintf(stderr, "\n  Choice [default=7]: ");
    std::fflush(stderr);

    int choice = 7;
    std::string line;
    if (std::getline(std::cin, line)) {
        try { if (!line.empty()) choice = std::stoi(line); } catch (...) {}
    }
    if (choice < 1 || choice > 7) choice = 7;

    use_enamine  = (choice == 1 || choice == 4 || choice == 5 || choice == 7);
    use_npass    = has_npass    && (choice == 2 || choice == 4 || choice == 6 || choice == 7);
    use_drugbank = has_drugbank && (choice == 3 || choice == 5 || choice == 6 || choice == 7);
    std::fprintf(stderr, "\n");
}

// ── Output mode ───────────────────────────────────────────────────────────────
enum class OutMode { COMBINED, PER_DB, BOTH };

static OutMode parse_out_mode(const std::string& s) {
    if (s == "per-db")   return OutMode::PER_DB;
    if (s == "both")     return OutMode::BOTH;
    return OutMode::COMBINED;
}

static void show_out_mode_menu(OutMode& mode) {
    std::fprintf(stderr, "  Select output format:\n");
    std::fprintf(stderr, "  [1] combined  — one TSV with all databases as columns\n");
    std::fprintf(stderr, "  [2] per-db    — separate TSV per database (matched rows only)\n");
    std::fprintf(stderr, "  [3] both      — combined AND per-db files\n");
    std::fprintf(stderr, "\n  Choice [1-3, default=1]: ");
    std::fflush(stderr);

    int choice = 1;
    std::string line;
    if (std::getline(std::cin, line)) {
        try { if (!line.empty()) choice = std::stoi(line); } catch (...) {}
    }
    if (choice < 1 || choice > 3) choice = 1;
    if (choice == 2) mode = OutMode::PER_DB;
    else if (choice == 3) mode = OutMode::BOTH;
    else mode = OutMode::COMBINED;
    std::fprintf(stderr, "\n");
}

// ── Output paths ──────────────────────────────────────────────────────────────
struct OutputPaths {
    std::string combined;
    std::string enamine;
    std::string npass;
    std::string drugbank;
};

static OutputPaths derive_paths(const std::string& base) {
    if (base.empty()) return {};
    fs::path p(base);
    std::string stem = p.stem().string();
    std::string dir  = p.parent_path().string();
    auto join = [&](const std::string& suffix) -> std::string {
        fs::path out = fs::path(dir.empty() ? "." : dir) / (stem + suffix + ".tsv");
        return out.string();
    };
    OutputPaths op;
    op.combined = base;
    op.enamine  = join("_enamine");
    op.npass    = join("_npass");
    op.drugbank = join("_drugbank");
    return op;
}

// For multiple queries in a session, number the output files to avoid overwriting.
static OutputPaths derive_paths_numbered(const std::string& base, int query_num) {
    if (base.empty()) return {};
    if (query_num == 0) return derive_paths(base);
    fs::path p(base);
    std::string stem = p.stem().string();
    std::string ext  = p.extension().string();
    std::string dir  = p.parent_path().string();
    char num[16]; std::snprintf(num, sizeof(num), "_q%03d", query_num + 1);
    std::string sfx = num;
    auto join = [&](const std::string& suf) -> std::string {
        return (fs::path(dir.empty() ? "." : dir) / (stem + sfx + suf + ext)).string();
    };
    OutputPaths op;
    op.combined = join("");
    op.enamine  = join("_enamine");
    op.npass    = join("_npass");
    op.drugbank = join("_drugbank");
    return op;
}

// ── Result types ──────────────────────────────────────────────────────────────
struct SearchResult {
    int      rank;
    uint64_t mol_id;
    float    faiss_dist;
    double   score;
    std::string   canon_smiles;
    EnamineRecord enamine;
    std::vector<NpassCompound>  npass_hits;
    std::vector<DrugBankDrug>   drugbank_hits;
};

struct DrugBankSimHit { double score; DrugBankDrug drug; };
struct NpassSimHit    { double score; NpassCompound compound; };

// ── Direct similarity search functions ────────────────────────────────────────
static std::vector<DrugBankSimHit> search_drugbank_sim(
        const DrugBankIndex& idx,
        const ExplicitBitVect& qfp,
        const FpConfig& fp_cfg,
        SimMetric metric,
        int top_n) {
    const auto& drugs = idx.drugs();
    int n = (int)drugs.size();
    std::vector<double> scores(n, -1.0);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; ++i) {
        if (drugs[i].smiles.empty()) continue;
        auto mol = smiles_to_mol(drugs[i].smiles);
        if (!mol) continue;
        auto fp = mol_to_fp_bv(*mol, fp_cfg);
        if (!fp) continue;
        scores[i] = compute_similarity(metric, qfp, *fp);
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<DrugBankSimHit> hits;
    hits.reserve(top_n);
    for (int i = 0; i < n && (int)hits.size() < top_n; ++i) {
        if (scores[order[i]] < 0.0) break;
        hits.push_back({scores[order[i]], drugs[order[i]]});
    }
    return hits;
}

static std::vector<NpassSimHit> search_npass_sim(
        const NpassIndex& npass,
        const ExplicitBitVect& qfp,
        const FpConfig& fp_cfg,
        SimMetric metric,
        int top_n) {
    auto all_smi = npass.all_smiles();
    int n = (int)all_smi.size();
    std::vector<double> scores(n, -1.0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        auto mol = smiles_to_mol(all_smi[i].second);
        if (!mol) continue;
        auto fp = mol_to_fp_bv(*mol, fp_cfg);
        if (!fp) continue;
        scores[i] = compute_similarity(metric, qfp, *fp);
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<NpassSimHit> hits;
    hits.reserve(top_n);
    for (int i = 0; i < n && (int)hits.size() < top_n; ++i) {
        if (scores[order[i]] < 0.0) break;
        NpassCompound c = npass.lookup_by_npid(all_smi[order[i]].first);
        hits.push_back({scores[order[i]], std::move(c)});
    }
    return hits;
}

// ── TSV cell sanitizer ────────────────────────────────────────────────────────
// Replaces chars that would break TSV or our record/field separators.
static std::string sanitize_cell(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        if (c == '\t' || c == '\n' || c == '\r') r += ' ';
        else if (c == '|') r += '/';   // field separator used in nested records
        else r += c;
    }
    return r;
}

// Join a string vector with a separator.
static std::string join_vec(const std::vector<std::string>& v, char sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) r += sep;
        r += v[i];
    }
    return r;
}

// Serialize one NpassRow as "key=value|key=value".
static std::string serialize_npass_row(const NpassRow& row, const std::string& prefix = {}) {
    std::string r;
    bool first = true;
    for (const auto& [k, v] : row) {
        if (!first) r += '|';
        first = false;
        r += prefix + k + '=' + sanitize_cell(v);
    }
    return r;
}

// Serialize a DrugBankTarget as "key=value|..." fields.
static std::string serialize_db_target(const DrugBankTarget& t) {
    std::string r;
    r += "name="              + sanitize_cell(t.name);
    r += "|uniprot="          + sanitize_cell(t.polypeptide_id);
    r += "|source="           + sanitize_cell(t.polypeptide_source);
    r += "|gene="             + sanitize_cell(t.gene_name);
    r += "|organism="         + sanitize_cell(t.organism);
    r += "|actions="          + sanitize_cell(join_vec(t.actions, ','));
    r += "|known_action="     + sanitize_cell(t.known_action);
    r += "|general_function=" + sanitize_cell(t.general_function);
    r += "|specific_function="+ sanitize_cell(t.specific_function);
    r += "|cellular_location="+ sanitize_cell(t.cellular_location);
    r += "|chromosome="       + sanitize_cell(t.chromosome_location);
    r += "|locus="            + sanitize_cell(t.locus);
    r += "|mw_pp="            + sanitize_cell(t.molecular_weight_pp);
    r += "|pi="               + sanitize_cell(t.theoretical_pi);
    return r;
}

// Serialize a vector of DrugBankTargets; records separated by ";;".
static std::string serialize_db_targets(const std::vector<DrugBankTarget>& tv) {
    std::string r;
    for (size_t i = 0; i < tv.size(); i++) {
        if (i) r += ";;";
        r += serialize_db_target(tv[i]);
    }
    return r;
}

// ── TSV writers ───────────────────────────────────────────────────────────────
static void write_combined_tsv_header(std::ostream& out,
                                       const std::vector<std::string>& enamine_hdr,
                                       bool use_enamine, bool use_npass, bool use_drugbank) {
    out << "rank\tmol_id\tscore\tfaiss_dist\tmetric\tquery_smiles\thit_smiles";
    if (use_enamine)
        for (const auto& col : enamine_hdr) out << '\t' << col;
    if (use_npass)
        out << "\tnpass_matched\tnpass_np_ids";
    if (use_drugbank)
        out << "\tdrugbank_matched\tdrugbank_db_ids";
    out << '\n';
}

static void write_combined_tsv_row(std::ostream& out, const SearchResult& r,
                                    const std::string& query_smiles,
                                    const std::string& metric_name,
                                    bool use_enamine, bool use_npass, bool use_drugbank) {
    out << r.rank << '\t' << r.mol_id << '\t'
        << std::fixed << std::setprecision(6) << r.score << '\t'
        << r.faiss_dist << '\t' << metric_name << '\t'
        << query_smiles << '\t' << r.canon_smiles;

    if (use_enamine)
        for (const auto& f : r.enamine.fields) out << '\t' << f;

    if (use_npass) {
        out << '\t' << (r.npass_hits.empty() ? "false" : "true") << '\t';
        for (size_t i = 0; i < r.npass_hits.size(); i++) {
            if (i) out << ';';
            out << r.npass_hits[i].np_id;
        }
    }

    if (use_drugbank) {
        out << '\t' << (r.drugbank_hits.empty() ? "false" : "true") << '\t';
        for (size_t i = 0; i < r.drugbank_hits.size(); i++) {
            if (i) out << ';';
            out << r.drugbank_hits[i].drugbank_id;
        }
    }
    out << '\n';
}

static void write_enamine_tsv_header(std::ostream& out,
                                      const std::vector<std::string>& enamine_hdr) {
    out << "rank\tmol_id\tscore\tfaiss_dist\tmetric\tquery_smiles\thit_smiles";
    for (const auto& col : enamine_hdr) out << '\t' << col;
    out << '\n';
}

static void write_enamine_tsv_row(std::ostream& out, const SearchResult& r,
                                   const std::string& query_smiles,
                                   const std::string& metric_name) {
    out << r.rank << '\t' << r.mol_id << '\t'
        << std::fixed << std::setprecision(6) << r.score << '\t'
        << r.faiss_dist << '\t' << metric_name << '\t'
        << query_smiles << '\t' << r.canon_smiles;
    for (const auto& f : r.enamine.fields) out << '\t' << f;
    out << '\n';
}

static void write_npass_tsv(std::ostream& out,
                             const std::vector<NpassSimHit>& hits,
                             const std::string& query_smiles,
                             const std::string& metric_name) {
    // Collect all generalinfo keys seen across hits (dynamic columns).
    std::vector<std::string> gi_keys;
    for (const auto& h : hits)
        for (const auto& [k, _] : h.compound.generalinfo)
            if (std::find(gi_keys.begin(), gi_keys.end(), k) == gi_keys.end())
                gi_keys.push_back(k);

    // Header
    out << "rank\tscore\tmetric\tquery_smiles"
           "\tnp_id\tinchi\tinchikey\tsmiles_npass";
    for (const auto& k : gi_keys) out << '\t' << k;
    out << "\tn_activities\tactivities_detail"
           "\tn_toxicities\ttoxicities_detail"
           "\tn_species\tspecies_detail"
           "\tn_coculture\tcoculture_detail"
           "\tn_elicitation\telicitation_detail"
           "\tn_engineer\tengineer_detail"
           "\tn_symbiont\tsymbiont_detail\n";

    // Helper: serialize a list of NpassRows, each as "key=value|...", records by ";;".
    auto write_rows = [&](const std::vector<NpassRow>& rows) {
        out << '\t' << rows.size() << '\t';
        for (size_t j = 0; j < rows.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(rows[j]);
        }
    };

    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        const auto& nc = h.compound;
        out << (i+1) << '\t'
            << std::fixed << std::setprecision(6) << h.score << '\t'
            << metric_name << '\t' << query_smiles << '\t'
            << nc.np_id << '\t'
            << sanitize_cell(nc.inchi) << '\t'
            << nc.inchikey << '\t'
            << sanitize_cell(nc.smiles_npass);

        // Dynamic generalinfo columns
        for (const auto& k : gi_keys) {
            out << '\t';
            auto it = nc.generalinfo.find(k);
            if (it != nc.generalinfo.end()) out << sanitize_cell(it->second);
        }

        // Activities: each ActivityRecord = activity row + joined target row
        out << '\t' << nc.activities.size() << '\t';
        for (size_t j = 0; j < nc.activities.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.activities[j].activity);
            if (!nc.activities[j].target.empty())
                out << '|' << serialize_npass_row(nc.activities[j].target, "target.");
        }

        // Toxicities: each ToxicityRecord = toxicity row + joined target row
        out << '\t' << nc.toxicities.size() << '\t';
        for (size_t j = 0; j < nc.toxicities.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.toxicities[j].toxicity);
            if (!nc.toxicities[j].target.empty())
                out << '|' << serialize_npass_row(nc.toxicities[j].target, "target.");
        }

        // Species pairs: species_pair row + joined species_info row
        out << '\t' << nc.species_pairs.size() << '\t';
        for (size_t j = 0; j < nc.species_pairs.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.species_pairs[j].species_pair);
            if (!nc.species_pairs[j].species_info.empty())
                out << '|' << serialize_npass_row(nc.species_pairs[j].species_info, "species.");
        }

        // Contextual biosynthesis tables
        write_rows(nc.coculture_records);
        write_rows(nc.elicitation_records);
        write_rows(nc.engineer_records);
        write_rows(nc.symbiont_records);

        out << '\n';
    }
}

static void write_drugbank_tsv(std::ostream& out,
                                const std::vector<DrugBankSimHit>& hits,
                                const std::string& query_smiles,
                                const std::string& metric_name) {
    // Header — every field from DrugBankDrug + DrugBankTarget
    out << "rank\tscore\tmetric\tquery_smiles"
           // Identifiers
           "\tdrugbank_id\talt_ids\tdrug_type\tname\tcas_number\tunii\tstate"
           // Lists
           "\tgroups\tsynonyms\tcategories"
           // Structure
           "\tsmiles_db\tinchikey_db\tinchi_db\tmolecular_formula\tmolecular_weight\tlogp"
           // Classification
           "\tclassif_kingdom\tclassif_superclass\tclassif_class\tclassif_subclass\tclassif_direct_parent"
           // Pharmacology text
           "\tdescription\tindication\tpharmacodynamics\tmechanism_of_action"
           "\ttoxicity_text\tmetabolism\tabsorption\thalf_life\tprotein_binding"
           "\troute_of_elimination\tvolume_of_distribution\tclearance\tsynthesis_reference"
           // Targets/enzymes/transporters/carriers (count + full detail)
           "\tn_targets\ttargets_detail"
           "\tn_enzymes\tenzymes_detail"
           "\tn_transporters\ttransporters_detail"
           "\tn_carriers\tcarriers_detail\n";

    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        const auto& d = h.drug;

        out << (i+1) << '\t'
            << std::fixed << std::setprecision(6) << h.score << '\t'
            << metric_name << '\t' << query_smiles << '\t'
            // Identifiers
            << d.drugbank_id << '\t'
            << sanitize_cell(join_vec(d.alt_ids, ';')) << '\t'
            << d.drug_type << '\t'
            << sanitize_cell(d.name) << '\t'
            << d.cas_number << '\t'
            << d.unii << '\t'
            << d.state << '\t'
            // Lists
            << join_vec(d.groups, ';') << '\t'
            << sanitize_cell(join_vec(d.synonyms, ';')) << '\t'
            << sanitize_cell(join_vec(d.categories, ';')) << '\t'
            // Structure
            << sanitize_cell(d.smiles) << '\t'
            << d.inchikey << '\t'
            << sanitize_cell(d.inchi) << '\t'
            << d.molecular_formula << '\t'
            << d.molecular_weight << '\t'
            << d.logp << '\t'
            // Classification
            << d.classif_kingdom << '\t'
            << d.classif_superclass << '\t'
            << d.classif_class << '\t'
            << d.classif_subclass << '\t'
            << d.classif_direct_parent << '\t'
            // Pharmacology text (long fields — sanitized)
            << sanitize_cell(d.description) << '\t'
            << sanitize_cell(d.indication) << '\t'
            << sanitize_cell(d.pharmacodynamics) << '\t'
            << sanitize_cell(d.mechanism_of_action) << '\t'
            << sanitize_cell(d.toxicity) << '\t'
            << sanitize_cell(d.metabolism) << '\t'
            << sanitize_cell(d.absorption) << '\t'
            << sanitize_cell(d.half_life) << '\t'
            << sanitize_cell(d.protein_binding) << '\t'
            << sanitize_cell(d.route_of_elimination) << '\t'
            << sanitize_cell(d.volume_of_distribution) << '\t'
            << sanitize_cell(d.clearance) << '\t'
            << sanitize_cell(d.synthesis_reference) << '\t'
            // Targets (count + serialized detail: each target as "key=value|...", records by ";;")
            << d.targets.size()      << '\t' << serialize_db_targets(d.targets)      << '\t'
            << d.enzymes.size()      << '\t' << serialize_db_targets(d.enzymes)      << '\t'
            << d.transporters.size() << '\t' << serialize_db_targets(d.transporters) << '\t'
            << d.carriers.size()     << '\t' << serialize_db_targets(d.carriers)
            << '\n';
    }
}

// ── Per-query profile ─────────────────────────────────────────────────────────
struct QueryProfile {
    double t_query_fp     = 0;
    double t_npass_sim    = 0;
    double t_drugbank_sim = 0;
    double t_faiss_search = 0;
    double t_retrieval    = 0;
    double t_fp_rerank    = 0;
    double t_sim_rerank   = 0;
    double t_annotation   = 0;
    double t_total        = 0;
    size_t n_retrieved    = 0;
    size_t n_reranked     = 0;
    bool   used_gpu       = false;
};

static void print_query_profile(const QueryProfile& p, const Config& cfg) {
    double t_rerank    = p.t_fp_rerank + p.t_sim_rerank;
    double pct_fp      = (t_rerank > 0) ? 100.0 * p.t_fp_rerank  / t_rerank : 0.0;
    double pct_sim     = (t_rerank > 0) ? 100.0 * p.t_sim_rerank / t_rerank : 0.0;
    double avg_read_ms = (p.n_retrieved > 0) ? p.t_retrieval / p.n_retrieved * 1000.0 : 0.0;

    struct Phase { const char* name; double t; };
    Phase phases[] = {
        {"Query FP",          p.t_query_fp},
        {"NPASS sim",         p.t_npass_sim},
        {"DrugBank sim",      p.t_drugbank_sim},
        {"FAISS search",      p.t_faiss_search},
        {"Enamine retrieval", p.t_retrieval},
        {"Rerank FP",         p.t_fp_rerank},
        {"Rerank sim",        p.t_sim_rerank},
        {"Annotation",        p.t_annotation},
    };
    std::sort(std::begin(phases), std::end(phases),
              [](const Phase& a, const Phase& b){ return a.t > b.t; });

    std::string bottleneck;
    for (int i = 0; i < 2 && i < (int)(sizeof(phases)/sizeof(phases[0])); i++) {
        if (phases[i].t < 0.001) break;
        if (!bottleneck.empty()) bottleneck += " + ";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s (%.1f%%)",
                      phases[i].name,
                      p.t_total > 0 ? 100.0 * phases[i].t / p.t_total : 0.0);
        bottleneck += buf;
    }

    std::fprintf(stderr,
        "\n[PROFILE] Query FP compute:        %7.4fs\n"
        "[PROFILE] NPASS sim search:         %7.4fs\n"
        "[PROFILE] DrugBank sim search:      %7.4fs\n"
        "[PROFILE] FAISS search (%s):      %7.4fs  (k=%d, nprobe=%d)\n"
        "[PROFILE] Enamine retrieval:        %7.4fs  (%zu mmap reads)\n"
        "[PROFILE]   └─ avg per read:        %7.4fms\n"
        "[PROFILE] Exact rerank (OMP):       %7.4fs  (%zu FP recompute)\n"
        "[PROFILE]   └─ FP compute:          %7.4fs  (%.1f%%)\n"
        "[PROFILE]   └─ similarity calc:     %7.4fs  (%.1f%%)\n"
        "[PROFILE] Annotation:               %7.4fs\n"
        "[PROFILE] ──────────────────────────────────────────\n"
        "[PROFILE] Query total:              %7.4fs\n"
        "[PROFILE] Bottleneck: %s\n\n",
        p.t_query_fp,
        p.t_npass_sim,
        p.t_drugbank_sim,
        p.used_gpu ? "GPU" : "CPU", p.t_faiss_search, cfg.candidate_k, cfg.nprobe,
        p.t_retrieval, p.n_retrieved,
        avg_read_ms,
        t_rerank, p.n_reranked,
        p.t_fp_rerank, pct_fp,
        p.t_sim_rerank, pct_sim,
        p.t_annotation,
        p.t_total,
        bottleneck.c_str());
}

// ── SearchSession ─────────────────────────────────────────────────────────────
struct SearchSession {
    Config        cfg;
    SimMetric     metric       = SimMetric::TANIMOTO;
    bool          use_gpu      = false;
    bool          use_enamine  = true;
    bool          use_npass    = false;
    bool          use_drugbank = false;
    OutMode       out_mode     = OutMode::COMBINED;

    faiss::Index* base_idx     = nullptr;
    EnamineReader enamine;
    NpassIndex    npass;
    bool          has_npass    = false;
    DrugBankIndex drugbank;
    bool          has_drugbank = false;
    size_t        index_bytes  = 0;

    SearchSession() = default;
    ~SearchSession() { delete base_idx; }
    SearchSession(const SearchSession&)            = delete;
    SearchSession& operator=(const SearchSession&) = delete;
};

// ── run_query: execute one search against the loaded session ──────────────────
static void run_query(SearchSession& sess, const std::string& raw_smiles, int query_num) {
    const Config& cfg    = sess.cfg;
    SimMetric     metric = sess.metric;
    auto t_query_start   = Clock::now();
    QueryProfile prof;
    prof.used_gpu = sess.use_gpu;

    // Derive output paths (numbered for queries after the first)
    OutputPaths opaths = derive_paths_numbered(cfg.out_path, query_num);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: parse and fingerprint query SMILES
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 1: parsing query SMILES\n");
    auto t_fp = Clock::now();

    auto qmol = smiles_to_mol(raw_smiles);
    if (!qmol) { LOG_ERR("Invalid query SMILES: %s", raw_smiles.c_str()); return; }
    std::string canon_q = mol_to_canonical_smiles(*qmol);

    auto qfp_bv = mol_to_fp_bv(*qmol, cfg.fp);
    if (!qfp_bv) { LOG_ERR("Fingerprint failed for query"); return; }

    std::vector<float> qvec(cfg.fp.nbits);
    mol_to_fp_float(*qmol, cfg.fp, qvec.data());
    prof.t_query_fp = elapsed_s(t_fp);

    std::fprintf(stderr, "  [OK]  Canonical: %s  |  FP: radius=%d nbits=%d  |  %.1f ms\n\n",
                 canon_q.c_str(), cfg.fp.radius, cfg.fp.nbits, prof.t_query_fp * 1000.0);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2 (per-query): NPASS similarity search
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<NpassSimHit> npass_sim_hits;
    if (sess.use_npass && sess.has_npass) {
        auto t_ns = Clock::now();
        std::fprintf(stderr, "[PHASE] Phase 2: NPASS similarity search\n");
        npass_sim_hits = search_npass_sim(sess.npass, *qfp_bv, cfg.fp, metric, cfg.top_n);
        prof.t_npass_sim = elapsed_s(t_ns);
        std::fprintf(stderr, "  [OK]  Top-%d hits  |  top score: %.4f  |  %.0f ms\n\n",
                     (int)npass_sim_hits.size(),
                     npass_sim_hits.empty() ? 0.0 : npass_sim_hits[0].score,
                     prof.t_npass_sim * 1000.0);
    } else {
        std::fprintf(stderr, "[PHASE] Phase 2: NPASS  ->  %s\n\n",
                     sess.use_npass ? "not loaded" : "skipped (not selected)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 3 (per-query): DrugBank similarity search
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<DrugBankSimHit> drugbank_sim_hits;
    if (sess.use_drugbank && sess.has_drugbank) {
        auto t_ds = Clock::now();
        std::fprintf(stderr, "[PHASE] Phase 3: DrugBank similarity search\n");
        drugbank_sim_hits = search_drugbank_sim(sess.drugbank, *qfp_bv, cfg.fp, metric, cfg.top_n);
        prof.t_drugbank_sim = elapsed_s(t_ds);
        std::fprintf(stderr, "  [OK]  Top-%d hits  |  top score: %.4f  |  %.0f ms\n\n",
                     (int)drugbank_sim_hits.size(),
                     drugbank_sim_hits.empty() ? 0.0 : drugbank_sim_hits[0].score,
                     prof.t_drugbank_sim * 1000.0);
    } else {
        std::fprintf(stderr, "[PHASE] Phase 3: DrugBank  ->  %s\n\n",
                     sess.use_drugbank ? "not loaded" : "skipped (not selected)");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 4: FAISS approximate nearest-neighbor search
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 4: FAISS search  (k=%d  nprobe=%d  backend=%s)\n",
                 cfg.candidate_k, cfg.nprobe, sess.use_gpu ? "GPU" : "CPU");
    auto t_search = Clock::now();

    std::vector<float>        D(cfg.candidate_k);
    std::vector<faiss::idx_t> I(cfg.candidate_k);

    if (sess.use_gpu) {
        std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;
        faiss::Index* gpu_idx = nullptr;
        try {
            gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
            gpu_idx = faiss::gpu::index_cpu_to_gpu(gpu_res.get(), 0, sess.base_idx);
            auto* gpu_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(gpu_idx);
            if (gpu_ivf) gpu_ivf->nprobe = (size_t)cfg.nprobe;
            gpu_idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
        } catch (const std::exception& e) {
            LOG_WARN("GPU search failed (%s), falling back to CPU", e.what());
            sess.base_idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
            prof.used_gpu = false;
        }
        if (gpu_idx) { delete gpu_idx; gpu_idx = nullptr; }
    } else {
        sess.base_idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
    }

    prof.t_faiss_search = elapsed_s(t_search);
    int valid_ids = 0;
    for (int i = 0; i < cfg.candidate_k; i++) if (I[i] >= 0) ++valid_ids;
    std::fprintf(stderr, "  [OK]  %d valid candidates returned  |  %.1f ms\n\n",
                 valid_ids, prof.t_faiss_search * 1000.0);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 5: retrieve Enamine records (serial mmap)
    // ─────────────────────────────────────────────────────────────────────────
    struct RawRecord {
        uint64_t      mol_id;
        float         faiss_dist;
        std::string   smiles;
        EnamineRecord rec;
    };
    std::vector<RawRecord> raw_records;
    raw_records.reserve(valid_ids);

    {
        Progress p5("Phase 5: retrieving Enamine records", (uint64_t)valid_ids, "candidates");
        auto t_ret = Clock::now();
        uint64_t processed = 0;

        for (int i = 0; i < cfg.candidate_k; i++) {
            if (I[i] < 0) { p5.update(++processed); continue; }
            uint64_t mid = static_cast<uint64_t>(I[i]);
            if (mid >= sess.enamine.count()) { p5.update(++processed); continue; }

            EnamineRecord rec = sess.enamine.fetch(mid);
            if (rec.fields.empty()) { p5.update(++processed); continue; }

            std::string smi = rec.smiles();
            if (smi.empty())  { p5.update(++processed); continue; }

            raw_records.push_back({mid, D[i], std::move(smi), std::move(rec)});
            p5.update(++processed);
        }
        p5.finish((uint64_t)valid_ids);
        prof.t_retrieval = elapsed_s(t_ret);
        prof.n_retrieved = raw_records.size();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 6: exact rerank — two OMP loops for clean timing
    //   Loop A: FP compute + canonical SMILES  → t_fp_rerank
    //   Loop B: similarity calc                → t_sim_rerank
    // ─────────────────────────────────────────────────────────────────────────
    int n_raw = (int)raw_records.size();
    using FpPtr = std::shared_ptr<ExplicitBitVect>;
    std::vector<FpPtr>       fps(n_raw);
    std::vector<std::string> canon_smiles(n_raw);
    std::vector<bool>        valid_fp(n_raw, false);

    {
        Progress p6a("Phase 6: computing rerank fingerprints", (uint64_t)n_raw, "mol");
        auto t_fp0 = Clock::now();

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n_raw; i++) {
            auto mol = smiles_to_mol(raw_records[i].smiles);
            if (!mol) continue;
            fps[i] = mol_to_fp_bv(*mol, cfg.fp);
            if (!fps[i]) continue;
            canon_smiles[i] = mol_to_canonical_smiles(*mol);
            valid_fp[i] = true;
            if (omp_get_thread_num() == 0)
                p6a.update((uint64_t)(i + 1));
        }
        p6a.finish((uint64_t)n_raw);
        prof.t_fp_rerank = elapsed_s(t_fp0);
    }

    std::vector<double> scores(n_raw, -1.0);
    {
        auto t_sim0 = Clock::now();
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n_raw; i++) {
            if (!valid_fp[i] || !fps[i]) continue;
            scores[i] = compute_similarity(metric, *qfp_bv, *fps[i]);
        }
        prof.t_sim_rerank = elapsed_s(t_sim0);
    }
    prof.n_reranked = (size_t)n_raw;

    // Collect and sort
    struct RawHit {
        uint64_t mol_id;
        float    faiss_dist;
        double   score;
        std::string   canon_smiles;
        EnamineRecord rec;
    };
    std::vector<RawHit> hits;
    hits.reserve(n_raw);
    for (int i = 0; i < n_raw; i++) {
        if (!valid_fp[i]) continue;
        hits.push_back({raw_records[i].mol_id, raw_records[i].faiss_dist,
                        scores[i], std::move(canon_smiles[i]), std::move(raw_records[i].rec)});
    }
    fps.clear();
    raw_records.clear();

    std::sort(hits.begin(), hits.end(),
              [](const RawHit& a, const RawHit& b) {
                  return a.score != b.score ? a.score > b.score
                                            : a.faiss_dist < b.faiss_dist;
              });

    std::fprintf(stderr, "\n  Valid hits after rerank: %s  (top score: %.4f)\n\n",
                 fmt_human(hits.size()).c_str(),
                 hits.empty() ? 0.0 : hits[0].score);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 7: annotate top-N hits with NPASS + DrugBank (Enamine-centric)
    // ─────────────────────────────────────────────────────────────────────────
    int top_n = std::min((int)hits.size(), cfg.top_n);
    std::vector<SearchResult> results;
    results.reserve(top_n);

    {
        auto t_ann = Clock::now();
        Progress p7("Phase 7: annotating hits", (uint64_t)top_n, "hits");

        for (int i = 0; i < top_n; i++) {
            SearchResult sr;
            sr.rank         = i + 1;
            sr.mol_id       = hits[i].mol_id;
            sr.faiss_dist   = hits[i].faiss_dist;
            sr.score        = hits[i].score;
            sr.canon_smiles = hits[i].canon_smiles;
            sr.enamine      = std::move(hits[i].rec);

            std::string ik = sr.enamine.inchikey();

            if (sess.has_npass)
                sr.npass_hits = sess.npass.lookup(ik, sr.canon_smiles);

            if (sess.has_drugbank) {
                const DrugBankDrug* db = sess.drugbank.lookup_by_inchikey(ik);
                if (!db) db = sess.drugbank.lookup_by_canon_smiles(sr.canon_smiles);
                if (db) sr.drugbank_hits.push_back(*db);
            }
            results.push_back(std::move(sr));
            p7.update((uint64_t)(i + 1));
        }
        p7.finish((uint64_t)top_n);
        prof.t_annotation = elapsed_s(t_ann);
    }

    int npass_matched    = 0;
    int drugbank_matched = 0;
    for (const auto& r : results) {
        if (!r.npass_hits.empty())    ++npass_matched;
        if (!r.drugbank_hits.empty()) ++drugbank_matched;
    }
    if (sess.has_npass)
        std::fprintf(stderr, "\n  NPASS matches   : %d / %d hits\n", npass_matched, top_n);
    if (sess.has_drugbank)
        std::fprintf(stderr, "  DrugBank matches: %d / %d hits\n", drugbank_matched, top_n);
    std::fprintf(stderr, "\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 8: print results to stdout
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 8: printing results\n");

    std::cout << "\n=== TOP " << top_n << " RESULTS"
              << "  query=" << canon_q
              << "  metric=" << cfg.metric_str << " ===\n";
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& r : results) {
        std::cout << r.rank
                  << ". mol_id=" << r.mol_id
                  << "  score=" << r.score
                  << "  dist=" << r.faiss_dist
                  << "  SMILES=" << r.canon_smiles;
        if (!r.npass_hits.empty()) {
            std::cout << "  NPASS=[";
            for (size_t ni = 0; ni < r.npass_hits.size(); ni++) {
                if (ni) std::cout << ',';
                std::cout << r.npass_hits[ni].np_id;
            }
            std::cout << ']';
        }
        if (!r.drugbank_hits.empty()) {
            std::cout << "  DrugBank=[";
            for (size_t di = 0; di < r.drugbank_hits.size(); di++) {
                if (di) std::cout << ',';
                std::cout << r.drugbank_hits[di].drugbank_id
                          << '(' << r.drugbank_hits[di].name << ')';
            }
            std::cout << ']';
        }
        std::cout << '\n';
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 9: write output files
    // ─────────────────────────────────────────────────────────────────────────
    if (!cfg.out_path.empty()) {
        std::fprintf(stderr, "\n[PHASE] Phase 9: writing output files\n");

        bool write_combined = (sess.out_mode == OutMode::COMBINED || sess.out_mode == OutMode::BOTH);
        bool write_per_db   = (sess.out_mode == OutMode::PER_DB   || sess.out_mode == OutMode::BOTH);

        if (write_combined) {
            auto t_w = Clock::now();
            std::ofstream tf(opaths.combined);
            write_combined_tsv_header(tf, sess.enamine.header(),
                                      sess.use_enamine, sess.use_npass, sess.use_drugbank);
            for (const auto& r : results)
                write_combined_tsv_row(tf, r, canon_q, cfg.metric_str,
                                       sess.use_enamine, sess.use_npass, sess.use_drugbank);
            std::fprintf(stderr, "  [OK]  Combined : %s  |  %.1f ms\n",
                         opaths.combined.c_str(), elapsed_s(t_w) * 1000.0);
        }

        if (write_per_db && sess.use_enamine) {
            auto t_w = Clock::now();
            std::ofstream tf(opaths.enamine);
            write_enamine_tsv_header(tf, sess.enamine.header());
            for (const auto& r : results)
                write_enamine_tsv_row(tf, r, canon_q, cfg.metric_str);
            std::fprintf(stderr, "  [OK]  Enamine  : %s  |  %.1f ms\n",
                         opaths.enamine.c_str(), elapsed_s(t_w) * 1000.0);
        }

        if (sess.use_npass && sess.has_npass && !npass_sim_hits.empty()) {
            auto t_w = Clock::now();
            std::ofstream tf(opaths.npass);
            write_npass_tsv(tf, npass_sim_hits, canon_q, cfg.metric_str);
            std::fprintf(stderr, "  [OK]  NPASS    : %s  (%d hits)  |  %.1f ms\n",
                         opaths.npass.c_str(), (int)npass_sim_hits.size(),
                         elapsed_s(t_w) * 1000.0);
        }

        if (sess.use_drugbank && sess.has_drugbank && !drugbank_sim_hits.empty()) {
            auto t_w = Clock::now();
            std::ofstream tf(opaths.drugbank);
            write_drugbank_tsv(tf, drugbank_sim_hits, canon_q, cfg.metric_str);
            std::fprintf(stderr, "  [OK]  DrugBank : %s  (%d hits)  |  %.1f ms\n",
                         opaths.drugbank.c_str(), (int)drugbank_sim_hits.size(),
                         elapsed_s(t_w) * 1000.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────────────
    prof.t_total = elapsed_s(t_query_start);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  Query         : %s\n",   canon_q.c_str());
    std::fprintf(stderr, "  Metric        : %s\n",   cfg.metric_str.c_str());
    std::fprintf(stderr, "  FAISS search  : %.1f ms\n", prof.t_faiss_search * 1000.0);
    std::fprintf(stderr, "  Query total   : %.1f ms\n", prof.t_total * 1000.0);
    std::fprintf(stderr, "  Enamine hits  : %d\n", top_n);
    if (sess.has_npass)
        std::fprintf(stderr, "  NPASS sim     : %d  (top score: %.4f)\n",
                     (int)npass_sim_hits.size(),
                     npass_sim_hits.empty() ? 0.0 : npass_sim_hits[0].score);
    if (sess.has_drugbank)
        std::fprintf(stderr, "  DrugBank sim  : %d  (top score: %.4f)\n",
                     (int)drugbank_sim_hits.size(),
                     drugbank_sim_hits.empty() ? 0.0 : drugbank_sim_hits[0].score);
    std::fprintf(stderr, "=================================================================\n");

    if (cfg.profile) print_query_profile(prof, cfg);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.faiss_path.empty() || cfg.enamine_path.empty() || cfg.offsets_path.empty()) {
        std::cerr <<
            "Usage: search_annotate\n"
            "  --faiss FILE --enamine FILE --enamine-offsets FILE\n"
            "  [--query SMILES] [--npass-index DIR] [--drugbank-bin FILE]\n"
            "  [--databases enamine,npass,drugbank] [--out-mode combined|per-db|both]\n"
            "  [--metric tanimoto|dice|tversky|cosine|kulczynski]\n"
            "  [--k 5000] [--top 50] [--nprobe 64] [--radius 3] [--nbits 4096]\n"
            "  [--out results.tsv] [--profile]\n";
        return 1;
    }
    if (cfg.header_path.empty())
        cfg.header_path = (fs::path(cfg.offsets_path).parent_path()
                           / "enamine.header.tsv").string();

    SearchSession sess;
    sess.cfg    = cfg;
    sess.metric = parse_metric(cfg.metric_str);
    sess.use_gpu = select_gpu_backend();

    // ── Database & output-mode selection (once per session) ───────────────────
    bool have_npass    = !cfg.npass_dir.empty()        && fs::exists(cfg.npass_dir);
    bool have_drugbank = !cfg.drugbank_bin_path.empty() && fs::exists(cfg.drugbank_bin_path);
    bool multi_db      = have_npass || have_drugbank;

    if (!cfg.databases_str.empty()) {
        parse_databases_flag(cfg.databases_str, sess.use_enamine, sess.use_npass, sess.use_drugbank);
        sess.use_npass    = sess.use_npass    && have_npass;
        sess.use_drugbank = sess.use_drugbank && have_drugbank;
    } else if (multi_db) {
        show_database_menu(sess.use_enamine, sess.use_npass, sess.use_drugbank, cfg);
    }

    if (!cfg.out_mode_str.empty()) {
        sess.out_mode = parse_out_mode(cfg.out_mode_str);
    } else if (!cfg.out_path.empty() && multi_db) {
        show_out_mode_menu(sess.out_mode);
    }

    // ── Session banner ────────────────────────────────────────────────────────
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  search_annotate  [SESSION MODE — databases loaded once]\n");
    std::fprintf(stderr, "  Databases : %s%s%s\n",
                 sess.use_enamine  ? "Enamine " : "",
                 sess.use_npass    ? "NPASS " : "",
                 sess.use_drugbank ? "DrugBank" : "");
    std::fprintf(stderr, "  k=%d  top=%d  nprobe=%d  radius=%d  nbits=%d\n",
                 cfg.candidate_k, cfg.top_n, cfg.nprobe, cfg.fp.radius, cfg.fp.nbits);
    std::fprintf(stderr, "  Backend   : %s\n",
                 sess.use_gpu ? "GPU (FAISS search)" : "CPU (FAISS search)");
    std::fprintf(stderr, "  Profile   : %s\n", cfg.profile ? "ON" : "OFF");
    std::fprintf(stderr, "=================================================================\n\n");

    // ── Session init: load all databases ──────────────────────────────────────
    auto t_init = Clock::now();

    // FAISS index
    {
        std::fprintf(stderr, "[INIT] Loading FAISS index\n");
        Spinner s_idx("Loading FAISS index");
        s_idx.tick(cfg.faiss_path.c_str());
        sess.base_idx = faiss::read_index(cfg.faiss_path.c_str());
        if (!sess.base_idx) {
            LOG_ERR("Cannot load FAISS index: %s", cfg.faiss_path.c_str());
            return 1;
        }
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(sess.base_idx);
        if (ivf) ivf->nprobe = cfg.nprobe;

        char summary[128];
        std::snprintf(summary, sizeof(summary), "%lld vectors  nprobe=%d",
                      (long long)sess.base_idx->ntotal, cfg.nprobe);
        s_idx.finish(summary);

        try { sess.index_bytes = (size_t)fs::file_size(cfg.faiss_path); } catch (...) {}
    }

    // Enamine mmap
    {
        std::fprintf(stderr, "\n[INIT] Opening Enamine mmap reader\n");
        auto t_mm = Clock::now();
        sess.enamine.open(cfg.enamine_path, cfg.offsets_path, cfg.header_path);
        std::fprintf(stderr, "  [OK]  %s molecules  |  %zu columns  |  %.1f ms\n",
                     fmt_human(sess.enamine.count()).c_str(),
                     sess.enamine.header().size(),
                     elapsed_s(t_mm) * 1000.0);
    }

    // NPASS
    if (sess.use_npass && !cfg.npass_dir.empty()) {
        std::fprintf(stderr, "\n[INIT] Loading NPASS index\n");
        auto t_np = Clock::now();
        std::string struct_in_npass = (fs::path(cfg.npass_dir) /
            "NPASS3.0_naturalproducts_structure.txt").string();
        std::string load_dir = fs::exists(struct_in_npass)
                               ? cfg.npass_dir
                               : (cfg.npass_raw_dir.empty() ? cfg.npass_dir : cfg.npass_raw_dir);
        try {
            sess.npass.load_from_dir(load_dir);
            sess.has_npass = true;
            std::fprintf(stderr, "  [OK]  %s compounds  |  %.2fs\n",
                         fmt_human(sess.npass.compound_count()).c_str(), elapsed_s(t_np));
        } catch (const std::exception& e) {
            LOG_WARN("NPASS load failed: %s", e.what());
        }
    }

    // DrugBank
    if (sess.use_drugbank && !cfg.drugbank_bin_path.empty()) {
        std::fprintf(stderr, "\n[INIT] Loading DrugBank index\n");
        auto t_db = Clock::now();
        Spinner s_db("Loading DrugBank binary index");
        s_db.tick(cfg.drugbank_bin_path.c_str());
        try {
            sess.drugbank.load_binary(cfg.drugbank_bin_path);
            sess.drugbank.build_smiles_index([](const std::string& smi) -> std::string {
                auto mol = smiles_to_mol(smi);
                if (!mol) return {};
                return mol_to_canonical_smiles(*mol);
            });
            sess.has_drugbank = true;
            char summary[64];
            std::snprintf(summary, sizeof(summary), "%s drugs  |  %.2fs",
                          fmt_human(sess.drugbank.drug_count()).c_str(), elapsed_s(t_db));
            s_db.finish(summary);
        } catch (const std::exception& e) {
            LOG_WARN("DrugBank load failed: %s", e.what());
            s_db.finish("FAILED");
        }
    }

    double init_s = elapsed_s(t_init);
    std::fprintf(stderr,
        "\n  [SESSION READY]  Init: %.2fs  |  FAISS: %zuMB  |  "
        "NPASS: %s  |  DrugBank: %s\n",
        init_s,
        sess.index_bytes / (1024 * 1024),
        sess.has_npass    ? "loaded" : "not loaded",
        sess.has_drugbank ? "loaded" : "not loaded");

    if (cfg.profile) {
        std::fprintf(stderr,
            "[PROFILE] Session init:             %7.2fs  (FAISS + Enamine + NPASS + DrugBank)\n",
            init_s);
    }

    // ── Query loop ─────────────────────────────────────────────────────────────
    std::string query_smiles = cfg.query_smiles;
    int query_num = 0;

    while (true) {
        // Get query SMILES (from CLI on first call if provided, else prompt)
        if (query_smiles.empty()) {
            std::fprintf(stderr, "\n  Enter query SMILES: ");
            std::fflush(stderr);
            if (!std::getline(std::cin, query_smiles) || query_smiles.empty()) break;
        }

        // Per-query mini-banner
        std::fprintf(stderr, "\n────────────────────────────────────────────────────────────────\n");
        std::fprintf(stderr, "  Query %d: %s\n", query_num + 1, query_smiles.c_str());
        std::fprintf(stderr, "────────────────────────────────────────────────────────────────\n\n");

        run_query(sess, query_smiles, query_num);

        // Prompt for another search
        std::fprintf(stderr, "\n  Do you want to perform another search? [Y/n]: ");
        std::fflush(stderr);
        std::string ans;
        if (!std::getline(std::cin, ans)) break;
        if (!ans.empty() && (ans[0] == 'n' || ans[0] == 'N')) break;

        // Get next SMILES
        query_smiles.clear();
        std::fprintf(stderr, "  Enter query SMILES: ");
        std::fflush(stderr);
        if (!std::getline(std::cin, query_smiles) || query_smiles.empty()) break;

        ++query_num;
    }

    std::fprintf(stderr, "\n  Session ended. All databases released.\n\n");
    return 0;
}
