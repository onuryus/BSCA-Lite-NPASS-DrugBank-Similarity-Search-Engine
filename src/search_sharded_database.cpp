// search_sharded_database — multi-shard similarity search across a sharded FAISS
// index built by build_sharded_database.
//
// All FAISS IDs are global molecule indices so a single EnamineReader serves all
// shards.  NPASS + DrugBank are loaded once per session.  For each query the tool
// iterates shards sequentially (or in parallel with --jobs N), collects per-shard
// top-K candidates, merges by FAISS distance, runs exact Tanimoto rerank, and
// annotates the final top hits.
//
// Usage:
//   search_sharded_database
//     --config  /data/index/shard_config.json
//     [--query  "SMILES"]         (first query; prompted if omitted)
//     [--npass-index  DIR]
//     [--drugbank-bin FILE]
//     [--databases    enamine,npass,drugbank]
//     [--out-mode     combined|per-db|both]
//     [--metric       tanimoto|dice|cosine]
//     [--k 200]                   (FAISS candidates per shard)
//     [--top 50]
//     [--nprobe 64]
//     [--jobs 1]                  (parallel shards; each job loads one FAISS index)
//     [--gpu | --no-gpu]
//     [--out results.tsv]
//     [--profile]

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
#include <mutex>
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
#include "common/shard_config.hpp"
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
    std::string config_path;
    std::string query_smiles;
    std::string npass_dir;
    std::string drugbank_bin_path;
    std::string databases_str;
    std::string out_mode_str;
    std::string out_path;
    std::string metric_str = "tanimoto";
    int    candidate_k  = 200;   // per-shard candidates
    int    top_n        = 50;
    int    nprobe       = 64;
    int    jobs         = 1;
    int    force_gpu    = -1;    // -1=ask, 0=CPU, 1=GPU
    FpConfig fp{3, 1024};
    bool   profile      = false;
};

static Config parse_args(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="--config"         &&i+1<argc) c.config_path        = argv[++i];
        else if (a=="--query"          &&i+1<argc) c.query_smiles       = argv[++i];
        else if (a=="--npass-index"    &&i+1<argc) c.npass_dir          = argv[++i];
        else if (a=="--drugbank-bin"   &&i+1<argc) c.drugbank_bin_path  = argv[++i];
        else if (a=="--databases"      &&i+1<argc) c.databases_str      = argv[++i];
        else if (a=="--out-mode"       &&i+1<argc) c.out_mode_str       = argv[++i];
        else if (a=="--out"            &&i+1<argc) c.out_path           = argv[++i];
        else if (a=="--metric"         &&i+1<argc) c.metric_str         = argv[++i];
        else if (a=="--k"              &&i+1<argc) c.candidate_k        = std::stoi(argv[++i]);
        else if (a=="--top"            &&i+1<argc) c.top_n              = std::stoi(argv[++i]);
        else if (a=="--nprobe"         &&i+1<argc) c.nprobe             = std::stoi(argv[++i]);
        else if (a=="--jobs"           &&i+1<argc) c.jobs               = std::stoi(argv[++i]);
        else if (a=="--gpu")                       c.force_gpu          = 1;
        else if (a=="--no-gpu")                    c.force_gpu          = 0;
        else if (a=="--radius"         &&i+1<argc) c.fp.radius          = std::stoi(argv[++i]);
        else if (a=="--nbits"          &&i+1<argc) c.fp.nbits           = std::stoi(argv[++i]);
        else if (a=="--profile")                   c.profile            = true;
    }
    return c;
}

// ── GPU/CPU selection ─────────────────────────────────────────────────────────
static bool select_gpu_backend(int force = -1) {
    if (force == 0) { std::fprintf(stderr, "  → CPU mode (forced)\n"); return false; }
    int num_gpus = 0;
    try { num_gpus = faiss::gpu::getNumDevices(); } catch (...) {}
    if (num_gpus == 0) {
        if (force == 1) std::fprintf(stderr, "  [WARN] --gpu requested but no GPU; using CPU\n");
        return false;
    }
    if (force == 1) { std::fprintf(stderr, "  → GPU mode (forced)\n"); return true; }

    std::string gpu_name = "Unknown";
    size_t vram_mb = 0;
    try {
        const cudaDeviceProp& prop = faiss::gpu::getDeviceProperties(0);
        gpu_name = prop.name;
        vram_mb  = prop.totalGlobalMem / (1024ULL * 1024ULL);
    } catch (...) {}

    std::fprintf(stderr, "\n  GPU detected: %s (%zuMB). Use GPU for search? [Y/n]: ",
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

// ── Database flag parsing ─────────────────────────────────────────────────────
static void parse_databases_flag(const std::string& s,
                                  bool& use_enamine, bool& use_npass, bool& use_drugbank) {
    use_enamine = use_npass = use_drugbank = false;
    std::string sl; for (char c : s) sl += (char)std::tolower((unsigned char)c);
    if (sl == "all") { use_enamine = use_npass = use_drugbank = true; return; }
    std::string tok;
    for (char c : sl + ",") {
        if (c == ',') {
            if (tok == "enamine")  use_enamine  = true;
            if (tok == "npass")    use_npass    = true;
            if (tok == "drugbank") use_drugbank = true;
            tok.clear();
        } else { tok += c; }
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
    if (s == "per-db") return OutMode::PER_DB;
    if (s == "both")   return OutMode::BOTH;
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
struct OutputPaths { std::string combined, enamine, npass, drugbank; };

static OutputPaths derive_paths(const std::string& base) {
    if (base.empty()) return {};
    fs::path p(base);
    std::string stem = p.stem().string();
    std::string dir  = p.parent_path().string();
    auto join = [&](const std::string& suf) {
        return (fs::path(dir.empty() ? "." : dir) / (stem + suf + ".tsv")).string();
    };
    return {base, join("_enamine"), join("_npass"), join("_drugbank")};
}

static OutputPaths derive_paths_numbered(const std::string& base, int qnum) {
    if (base.empty()) return {};
    if (qnum == 0) return derive_paths(base);
    fs::path p(base);
    std::string stem = p.stem().string(), ext = p.extension().string();
    std::string dir  = p.parent_path().string();
    char num[16]; std::snprintf(num, sizeof(num), "_q%03d", qnum + 1);
    std::string sfx = num;
    auto join = [&](const std::string& suf) {
        return (fs::path(dir.empty() ? "." : dir) / (stem + sfx + suf + ext)).string();
    };
    return {join(""), join("_enamine"), join("_npass"), join("_drugbank")};
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

// ── Direct DB similarity search ───────────────────────────────────────────────
static std::vector<DrugBankSimHit> search_drugbank_sim(
        const DrugBankIndex& idx, const ExplicitBitVect& qfp,
        const FpConfig& fp_cfg, SimMetric metric, int top_n) {
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
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });
    std::vector<DrugBankSimHit> hits;
    for (int i = 0; i < n && (int)hits.size() < top_n; ++i)
        if (scores[order[i]] >= 0.0)
            hits.push_back({scores[order[i]], drugs[order[i]]});
    return hits;
}

static std::vector<NpassSimHit> search_npass_sim(
        const NpassIndex& npass, const ExplicitBitVect& qfp,
        const FpConfig& fp_cfg, SimMetric metric, int top_n) {
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
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });
    std::vector<NpassSimHit> hits;
    for (int i = 0; i < n && (int)hits.size() < top_n; ++i) {
        if (scores[order[i]] < 0.0) break;
        NpassCompound c = npass.lookup_by_npid(all_smi[order[i]].first);
        hits.push_back({scores[order[i]], std::move(c)});
    }
    return hits;
}

// ── TSV helpers ───────────────────────────────────────────────────────────────
static std::string sanitize_cell(const std::string& s) {
    std::string r; r.reserve(s.size());
    for (char c : s) {
        if (c=='\t'||c=='\n'||c=='\r') r += ' ';
        else if (c=='|') r += '/';
        else r += c;
    }
    return r;
}
static std::string join_vec(const std::vector<std::string>& v, char sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) { if (i) r += sep; r += v[i]; }
    return r;
}
static std::string serialize_npass_row(const NpassRow& row, const std::string& prefix = {}) {
    std::string r; bool first = true;
    for (const auto& [k, v] : row) {
        if (!first) r += '|'; first = false;
        r += prefix + k + '=' + sanitize_cell(v);
    }
    return r;
}
static std::string serialize_db_target(const DrugBankTarget& t) {
    return "name=" + sanitize_cell(t.name)
        + "|uniprot=" + sanitize_cell(t.polypeptide_id)
        + "|source="  + sanitize_cell(t.polypeptide_source)
        + "|gene="    + sanitize_cell(t.gene_name)
        + "|organism="+ sanitize_cell(t.organism)
        + "|actions=" + sanitize_cell(join_vec(t.actions, ','))
        + "|known_action=" + sanitize_cell(t.known_action)
        + "|general_function=" + sanitize_cell(t.general_function)
        + "|specific_function="+ sanitize_cell(t.specific_function)
        + "|cellular_location="+ sanitize_cell(t.cellular_location)
        + "|chromosome=" + sanitize_cell(t.chromosome_location)
        + "|locus="      + sanitize_cell(t.locus)
        + "|mw_pp="      + sanitize_cell(t.molecular_weight_pp)
        + "|pi="         + sanitize_cell(t.theoretical_pi);
}
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
    if (use_enamine) for (const auto& col : enamine_hdr) out << '\t' << col;
    if (use_npass)    out << "\tnpass_matched\tnpass_np_ids";
    if (use_drugbank) out << "\tdrugbank_matched\tdrugbank_db_ids";
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
    if (use_enamine) for (const auto& f : r.enamine.fields) out << '\t' << f;
    if (use_npass) {
        out << '\t' << (r.npass_hits.empty() ? "false" : "true") << '\t';
        for (size_t i = 0; i < r.npass_hits.size(); i++) {
            if (i) out << ';'; out << r.npass_hits[i].np_id;
        }
    }
    if (use_drugbank) {
        out << '\t' << (r.drugbank_hits.empty() ? "false" : "true") << '\t';
        for (size_t i = 0; i < r.drugbank_hits.size(); i++) {
            if (i) out << ';'; out << r.drugbank_hits[i].drugbank_id;
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
    std::vector<std::string> gi_keys;
    for (const auto& h : hits)
        for (const auto& [k, _] : h.compound.generalinfo)
            if (std::find(gi_keys.begin(), gi_keys.end(), k) == gi_keys.end())
                gi_keys.push_back(k);

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

    auto write_rows = [&](const std::vector<NpassRow>& rows) {
        out << '\t' << rows.size() << '\t';
        for (size_t j = 0; j < rows.size(); j++) {
            if (j) out << ";;"; out << serialize_npass_row(rows[j]);
        }
    };
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i]; const auto& nc = h.compound;
        out << (i+1) << '\t' << std::fixed << std::setprecision(6) << h.score << '\t'
            << metric_name << '\t' << query_smiles << '\t'
            << nc.np_id << '\t' << sanitize_cell(nc.inchi) << '\t'
            << nc.inchikey << '\t' << sanitize_cell(nc.smiles_npass);
        for (const auto& k : gi_keys) {
            out << '\t';
            auto it = nc.generalinfo.find(k);
            if (it != nc.generalinfo.end()) out << sanitize_cell(it->second);
        }
        out << '\t' << nc.activities.size() << '\t';
        for (size_t j = 0; j < nc.activities.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.activities[j].activity);
            if (!nc.activities[j].target.empty())
                out << '|' << serialize_npass_row(nc.activities[j].target, "target.");
        }
        out << '\t' << nc.toxicities.size() << '\t';
        for (size_t j = 0; j < nc.toxicities.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.toxicities[j].toxicity);
            if (!nc.toxicities[j].target.empty())
                out << '|' << serialize_npass_row(nc.toxicities[j].target, "target.");
        }
        out << '\t' << nc.species_pairs.size() << '\t';
        for (size_t j = 0; j < nc.species_pairs.size(); j++) {
            if (j) out << ";;";
            out << serialize_npass_row(nc.species_pairs[j].species_pair);
            if (!nc.species_pairs[j].species_info.empty())
                out << '|' << serialize_npass_row(nc.species_pairs[j].species_info, "species.");
        }
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
    out << "rank\tscore\tmetric\tquery_smiles"
           "\tdrugbank_id\talt_ids\tdrug_type\tname\tcas_number\tunii\tstate"
           "\tgroups\tsynonyms\tcategories"
           "\tsmiles_db\tinchikey_db\tinchi_db\tmolecular_formula\tmolecular_weight\tlogp"
           "\tclassif_kingdom\tclassif_superclass\tclassif_class\tclassif_subclass\tclassif_direct_parent"
           "\tdescription\tindication\tpharmacodynamics\tmechanism_of_action"
           "\ttoxicity_text\tmetabolism\tabsorption\thalf_life\tprotein_binding"
           "\troute_of_elimination\tvolume_of_distribution\tclearance\tsynthesis_reference"
           "\tn_targets\ttargets_detail"
           "\tn_enzymes\tenzymes_detail"
           "\tn_transporters\ttransporters_detail"
           "\tn_carriers\tcarriers_detail\n";

    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i]; const auto& d = h.drug;
        out << (i+1) << '\t' << std::fixed << std::setprecision(6) << h.score << '\t'
            << metric_name << '\t' << query_smiles << '\t'
            << d.drugbank_id << '\t' << sanitize_cell(join_vec(d.alt_ids, ';')) << '\t'
            << d.drug_type << '\t' << sanitize_cell(d.name) << '\t'
            << d.cas_number << '\t' << d.unii << '\t' << d.state << '\t'
            << join_vec(d.groups, ';') << '\t'
            << sanitize_cell(join_vec(d.synonyms, ';')) << '\t'
            << sanitize_cell(join_vec(d.categories, ';')) << '\t'
            << sanitize_cell(d.smiles) << '\t'
            << d.inchikey << '\t' << sanitize_cell(d.inchi) << '\t'
            << d.molecular_formula << '\t' << d.molecular_weight << '\t' << d.logp << '\t'
            << d.classif_kingdom << '\t' << d.classif_superclass << '\t'
            << d.classif_class   << '\t' << d.classif_subclass   << '\t'
            << d.classif_direct_parent << '\t'
            << sanitize_cell(d.description)        << '\t'
            << sanitize_cell(d.indication)         << '\t'
            << sanitize_cell(d.pharmacodynamics)   << '\t'
            << sanitize_cell(d.mechanism_of_action)<< '\t'
            << sanitize_cell(d.toxicity)           << '\t'
            << sanitize_cell(d.metabolism)         << '\t'
            << sanitize_cell(d.absorption)         << '\t'
            << sanitize_cell(d.half_life)          << '\t'
            << sanitize_cell(d.protein_binding)    << '\t'
            << sanitize_cell(d.route_of_elimination) << '\t'
            << sanitize_cell(d.volume_of_distribution) << '\t'
            << sanitize_cell(d.clearance)          << '\t'
            << sanitize_cell(d.synthesis_reference)<< '\t'
            << d.targets.size()      << '\t' << serialize_db_targets(d.targets)      << '\t'
            << d.enzymes.size()      << '\t' << serialize_db_targets(d.enzymes)      << '\t'
            << d.transporters.size() << '\t' << serialize_db_targets(d.transporters) << '\t'
            << d.carriers.size()     << '\t' << serialize_db_targets(d.carriers)
            << '\n';
    }
}

// ── Session ───────────────────────────────────────────────────────────────────
struct Session {
    Config       cfg;
    ShardConfig  shards;
    SimMetric    metric       = SimMetric::TANIMOTO;
    bool         use_gpu      = false;
    bool         use_enamine  = true;
    bool         use_npass    = false;
    bool         use_drugbank = false;
    OutMode      out_mode     = OutMode::COMBINED;
    EnamineReader enamine;
    NpassIndex    npass;
    bool          has_npass    = false;
    DrugBankIndex drugbank;
    bool          has_drugbank = false;
};

// ── Per-shard search result (candidates before rerank) ────────────────────────
struct ShardCandidate {
    uint64_t mol_id;
    float    faiss_dist;
};

// ── run_query ─────────────────────────────────────────────────────────────────
static void run_query(Session& sess, const std::string& raw_smiles, int query_num) {
    const Config& cfg  = sess.cfg;
    SimMetric     metric = sess.metric;
    auto t_query_start  = Clock::now();

    OutputPaths opaths = derive_paths_numbered(cfg.out_path, query_num);

    // ── Phase 1: parse query SMILES ───────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 1: parsing query SMILES\n");
    auto t_fp = Clock::now();

    auto qmol = smiles_to_mol(raw_smiles);
    if (!qmol) { LOG_ERR("Invalid query SMILES: %s", raw_smiles.c_str()); return; }
    std::string canon_q = mol_to_canonical_smiles(*qmol);

    auto qfp_bv = mol_to_fp_bv(*qmol, cfg.fp);
    if (!qfp_bv) { LOG_ERR("Fingerprint failed for query"); return; }
    std::vector<float> qvec(cfg.fp.nbits);
    mol_to_fp_float(*qmol, cfg.fp, qvec.data());

    double t_phase1 = elapsed_s(t_fp);
    std::fprintf(stderr, "  [OK]  Canonical: %s  |  FP: radius=%d nbits=%d  |  %.1f ms\n\n",
                 canon_q.c_str(), cfg.fp.radius, cfg.fp.nbits, t_phase1 * 1000.0);

    // ── Phase 2: NPASS sim search ─────────────────────────────────────────────
    std::vector<NpassSimHit> npass_sim_hits;
    if (sess.use_npass && sess.has_npass) {
        auto t_ns = Clock::now();
        std::fprintf(stderr, "[PHASE] Phase 2: NPASS similarity search\n");
        npass_sim_hits = search_npass_sim(sess.npass, *qfp_bv, cfg.fp, metric, cfg.top_n);
        std::fprintf(stderr, "  [OK]  %d hits  |  top: %.4f  |  %.0f ms\n\n",
                     (int)npass_sim_hits.size(),
                     npass_sim_hits.empty() ? 0.0 : npass_sim_hits[0].score,
                     elapsed_s(t_ns) * 1000.0);
    } else {
        std::fprintf(stderr, "[PHASE] Phase 2: NPASS  →  %s\n\n",
                     sess.use_npass ? "not loaded" : "skipped");
    }

    // ── Phase 3: DrugBank sim search ──────────────────────────────────────────
    std::vector<DrugBankSimHit> drugbank_sim_hits;
    if (sess.use_drugbank && sess.has_drugbank) {
        auto t_ds = Clock::now();
        std::fprintf(stderr, "[PHASE] Phase 3: DrugBank similarity search\n");
        drugbank_sim_hits = search_drugbank_sim(sess.drugbank, *qfp_bv, cfg.fp, metric, cfg.top_n);
        std::fprintf(stderr, "  [OK]  %d hits  |  top: %.4f  |  %.0f ms\n\n",
                     (int)drugbank_sim_hits.size(),
                     drugbank_sim_hits.empty() ? 0.0 : drugbank_sim_hits[0].score,
                     elapsed_s(t_ds) * 1000.0);
    } else {
        std::fprintf(stderr, "[PHASE] Phase 3: DrugBank  →  %s\n\n",
                     sess.use_drugbank ? "not loaded" : "skipped");
    }

    // ── Phase 4: multi-shard FAISS search ────────────────────────────────────
    size_t num_shards = sess.shards.shards.size();
    std::fprintf(stderr, "[PHASE] Phase 4: searching %zu shard(s)  "
                         "(k=%d  nprobe=%d  jobs=%d)\n",
                 num_shards, cfg.candidate_k, cfg.nprobe, cfg.jobs);
    auto t_shard_start = Clock::now();

    // All candidates from all shards pooled here (thread-safe via mutex)
    std::vector<ShardCandidate> all_candidates;
    all_candidates.reserve((size_t)cfg.candidate_k * num_shards);
    std::mutex cand_mu;

    int jobs = std::max(1, std::min(cfg.jobs, (int)num_shards));

    // Process shards in batches of `jobs` — within each batch, open/search/close in parallel
    for (size_t base = 0; base < num_shards; base += (size_t)jobs) {
        size_t batch_end = std::min(base + (size_t)jobs, num_shards);
        size_t batch_sz  = batch_end - base;

        // Pre-load FAISS indices for this batch
        std::vector<faiss::Index*> batch_indices(batch_sz, nullptr);
        std::vector<bool>          load_ok(batch_sz, false);

        // Sequential load to avoid CUDA context conflicts from multiple threads
        for (size_t bi = 0; bi < batch_sz; bi++) {
            const auto& sh = sess.shards.shards[base + bi];
            if (!fs::exists(sh.faiss_path)) {
                std::fprintf(stderr, "  [WARN] shard %d: %s not found, skipping\n",
                             sh.id + 1, sh.faiss_path.c_str());
                continue;
            }
            try {
                batch_indices[bi] = faiss::read_index(sh.faiss_path.c_str());
                load_ok[bi] = true;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  [WARN] shard %d load failed: %s\n",
                             sh.id + 1, e.what());
            }
        }

        // Search all shards in this batch (one thread per shard via OMP)
        #pragma omp parallel for num_threads(jobs) schedule(static,1)
        for (int bi = 0; bi < (int)batch_sz; bi++) {
            if (!load_ok[bi] || !batch_indices[bi]) continue;
            const auto& sh = sess.shards.shards[base + (size_t)bi];

            faiss::Index* idx = batch_indices[bi];
            std::vector<float>        D(cfg.candidate_k, std::numeric_limits<float>::max());
            std::vector<faiss::idx_t> I(cfg.candidate_k, -1);

            // Set nprobe on IVF indices
            if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx))
                ivf->nprobe = (size_t)cfg.nprobe;

            try {
                if (sess.use_gpu) {
                    faiss::gpu::StandardGpuResources gpu_res;
                    faiss::Index* gpu_idx = nullptr;
                    try {
                        gpu_idx = faiss::gpu::index_cpu_to_gpu(&gpu_res, 0, idx);
                        if (auto* g = dynamic_cast<faiss::gpu::GpuIndexIVF*>(gpu_idx))
                            g->nprobe = (size_t)cfg.nprobe;
                        gpu_idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
                        delete gpu_idx;
                    } catch (...) {
                        if (gpu_idx) { delete gpu_idx; gpu_idx = nullptr; }
                        // Fall back to CPU
                        idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
                    }
                } else {
                    idx->search(1, qvec.data(), cfg.candidate_k, D.data(), I.data());
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  [WARN] shard %d search error: %s\n",
                             sh.id + 1, e.what());
            }

            // Collect valid results
            std::vector<ShardCandidate> local;
            local.reserve(cfg.candidate_k);
            for (int ci = 0; ci < cfg.candidate_k; ci++) {
                if (I[ci] < 0) continue;
                local.push_back({static_cast<uint64_t>(I[ci]), D[ci]});
            }

            {
                std::lock_guard<std::mutex> lk(cand_mu);
                all_candidates.insert(all_candidates.end(), local.begin(), local.end());
            }
        }

        // Free loaded indices
        for (size_t bi = 0; bi < batch_sz; bi++) {
            delete batch_indices[bi];
            batch_indices[bi] = nullptr;
        }

        size_t done = batch_end;
        std::fprintf(stderr, "  Shards searched: %zu / %zu  (pool: %zu candidates)\r",
                     done, num_shards, all_candidates.size());
        std::fflush(stderr);
    }
    std::fprintf(stderr, "\n");

    double t_shard_total = elapsed_s(t_shard_start);

    // Global merge: sort by FAISS distance, deduplicate by mol_id, keep top candidate_k
    {
        std::stable_sort(all_candidates.begin(), all_candidates.end(),
                  [](const ShardCandidate& a, const ShardCandidate& b) {
                      return a.faiss_dist < b.faiss_dist;
                  });
        // Deduplicate: same mol_id may appear from different shards (shouldn't in
        // a properly built shard set, but guard anyway)
        std::vector<ShardCandidate> deduped;
        deduped.reserve(all_candidates.size());
        uint64_t last_id = UINT64_MAX;
        for (auto& c : all_candidates) {
            if (c.mol_id != last_id) { deduped.push_back(c); last_id = c.mol_id; }
        }
        all_candidates = std::move(deduped);
    }

    int merged_k = std::min((int)all_candidates.size(), cfg.candidate_k * (int)num_shards);
    std::fprintf(stderr, "  [OK]  %d merged candidates  |  %.1f ms  (%.1f ms/shard)\n\n",
                 (int)all_candidates.size(), t_shard_total * 1000.0,
                 num_shards > 0 ? t_shard_total / (double)num_shards * 1000.0 : 0.0);
    (void)merged_k;

    // ── Phase 5: retrieve Enamine records ─────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 5: retrieving Enamine records (%zu candidates)\n",
                 all_candidates.size());
    auto t_ret = Clock::now();

    struct RawRecord {
        uint64_t mol_id; float faiss_dist; std::string smiles; EnamineRecord rec;
    };
    std::vector<RawRecord> raw_records;
    raw_records.reserve(all_candidates.size());

    for (const auto& cand : all_candidates) {
        if (cand.mol_id >= sess.enamine.count()) continue;
        EnamineRecord rec = sess.enamine.fetch(cand.mol_id);
        if (rec.fields.empty()) continue;
        std::string smi = rec.smiles();
        if (smi.empty()) continue;
        raw_records.push_back({cand.mol_id, cand.faiss_dist, std::move(smi), std::move(rec)});
    }
    double t_retrieval = elapsed_s(t_ret);
    std::fprintf(stderr, "  [OK]  %zu valid records  |  %.1f ms\n\n",
                 raw_records.size(), t_retrieval * 1000.0);

    // ── Phase 6: exact rerank (OMP) ───────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 6: exact Tanimoto rerank\n");
    int n_raw = (int)raw_records.size();
    using FpPtr = std::shared_ptr<ExplicitBitVect>;
    std::vector<FpPtr>       fps(n_raw);
    std::vector<std::string> canon_smiles(n_raw);
    std::vector<bool>        valid_fp(n_raw, false);

    {
        auto t_fp0 = Clock::now();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n_raw; i++) {
            auto mol = smiles_to_mol(raw_records[i].smiles);
            if (!mol) continue;
            fps[i] = mol_to_fp_bv(*mol, cfg.fp);
            if (!fps[i]) continue;
            canon_smiles[i] = mol_to_canonical_smiles(*mol);
            valid_fp[i] = true;
        }
        std::fprintf(stderr, "  [OK]  FP compute  |  %.1f ms\n", elapsed_s(t_fp0) * 1000.0);
    }

    std::vector<double> scores(n_raw, -1.0);
    {
        auto t_sim0 = Clock::now();
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n_raw; i++) {
            if (!valid_fp[i] || !fps[i]) continue;
            scores[i] = compute_similarity(metric, *qfp_bv, *fps[i]);
        }
        std::fprintf(stderr, "  [OK]  Similarity  |  %.1f ms\n", elapsed_s(t_sim0) * 1000.0);
    }

    struct RawHit {
        uint64_t mol_id; float faiss_dist; double score;
        std::string canon_smiles; EnamineRecord rec;
    };
    std::vector<RawHit> hits;
    hits.reserve(n_raw);
    for (int i = 0; i < n_raw; i++) {
        if (!valid_fp[i]) continue;
        hits.push_back({raw_records[i].mol_id, raw_records[i].faiss_dist,
                        scores[i], std::move(canon_smiles[i]), std::move(raw_records[i].rec)});
    }
    fps.clear(); raw_records.clear();

    std::sort(hits.begin(), hits.end(),
              [](const RawHit& a, const RawHit& b) {
                  return a.score != b.score ? a.score > b.score
                                            : a.faiss_dist < b.faiss_dist;
              });

    std::fprintf(stderr, "\n  Valid hits after rerank: %s  (top score: %.4f)\n\n",
                 fmt_human(hits.size()).c_str(),
                 hits.empty() ? 0.0 : hits[0].score);

    // ── Phase 7: annotate top-N ───────────────────────────────────────────────
    int top_n = std::min((int)hits.size(), cfg.top_n);
    std::vector<SearchResult> results;
    results.reserve(top_n);

    {
        auto t_ann = Clock::now();
        std::fprintf(stderr, "[PHASE] Phase 7: annotating %d hits\n", top_n);
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
        }
        std::fprintf(stderr, "  [OK]  Annotation  |  %.1f ms\n\n", elapsed_s(t_ann) * 1000.0);
    }

    int npass_matched = 0, drugbank_matched = 0;
    for (const auto& r : results) {
        if (!r.npass_hits.empty())    ++npass_matched;
        if (!r.drugbank_hits.empty()) ++drugbank_matched;
    }
    if (sess.has_npass)    std::fprintf(stderr, "  NPASS matches   : %d / %d\n", npass_matched, top_n);
    if (sess.has_drugbank) std::fprintf(stderr, "  DrugBank matches: %d / %d\n", drugbank_matched, top_n);
    std::fprintf(stderr, "\n");

    // ── Phase 8: print results ────────────────────────────────────────────────
    std::string metric_name = cfg.metric_str;

    // Stdout summary
    std::cout << "\n=== TOP " << top_n << " RESULTS"
              << "  query=" << canon_q
              << "  shards=" << num_shards << " ===\n";
    for (const auto& r : results) {
        std::cout << r.rank << "\t" << std::fixed << std::setprecision(4) << r.score
                  << "\t" << r.mol_id << "\t" << r.canon_smiles;
        if (!r.npass_hits.empty())    std::cout << "\t[NPASS:" << r.npass_hits[0].np_id << "]";
        if (!r.drugbank_hits.empty()) std::cout << "\t[DB:" << r.drugbank_hits[0].drugbank_id << "]";
        std::cout << "\n";
    }
    std::cout << "\n";

    // ── Phase 9: write TSV ────────────────────────────────────────────────────
    if (opaths.combined.empty()) {
        double t_total = elapsed_s(t_query_start);
        std::fprintf(stderr, "  Query total: %.3f s\n\n", t_total);
        return;
    }

    std::fprintf(stderr, "[PHASE] Phase 9: writing TSV output\n");
    auto& ehdr = sess.enamine.header();

    auto write_combined = [&](const std::string& path) {
        std::ofstream out(path);
        if (!out) { std::fprintf(stderr, "  [WARN] Cannot open %s\n", path.c_str()); return; }
        write_combined_tsv_header(out, ehdr, sess.use_enamine, sess.use_npass, sess.use_drugbank);
        for (const auto& r : results)
            write_combined_tsv_row(out, r, canon_q, metric_name,
                                   sess.use_enamine, sess.use_npass, sess.use_drugbank);
        std::fprintf(stderr, "  Combined TSV : %s\n", path.c_str());
    };
    auto write_enamine_file = [&](const std::string& path) {
        std::ofstream out(path);
        if (!out) { std::fprintf(stderr, "  [WARN] Cannot open %s\n", path.c_str()); return; }
        write_enamine_tsv_header(out, ehdr);
        for (const auto& r : results) write_enamine_tsv_row(out, r, canon_q, metric_name);
        std::fprintf(stderr, "  Enamine TSV  : %s\n", path.c_str());
    };
    auto write_npass_file = [&](const std::string& path) {
        std::ofstream out(path);
        if (!out) { std::fprintf(stderr, "  [WARN] Cannot open %s\n", path.c_str()); return; }
        write_npass_tsv(out, npass_sim_hits, canon_q, metric_name);
        std::fprintf(stderr, "  NPASS TSV    : %s\n", path.c_str());
    };
    auto write_drugbank_file = [&](const std::string& path) {
        std::ofstream out(path);
        if (!out) { std::fprintf(stderr, "  [WARN] Cannot open %s\n", path.c_str()); return; }
        write_drugbank_tsv(out, drugbank_sim_hits, canon_q, metric_name);
        std::fprintf(stderr, "  DrugBank TSV : %s\n", path.c_str());
    };

    if (sess.out_mode == OutMode::COMBINED || sess.out_mode == OutMode::BOTH)
        write_combined(opaths.combined);
    if (sess.out_mode == OutMode::PER_DB || sess.out_mode == OutMode::BOTH) {
        if (sess.use_enamine)  write_enamine_file(opaths.enamine);
        if (sess.use_npass && !npass_sim_hits.empty())    write_npass_file(opaths.npass);
        if (sess.use_drugbank && !drugbank_sim_hits.empty()) write_drugbank_file(opaths.drugbank);
    }

    double t_total = elapsed_s(t_query_start);
    std::fprintf(stderr, "\n  Query total: %.3f s\n\n", t_total);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.config_path.empty()) {
        std::fprintf(stderr,
            "Usage: search_sharded_database --config shard_config.json [options]\n"
            "  [--query SMILES] [--npass-index DIR] [--drugbank-bin FILE]\n"
            "  [--databases enamine,npass,drugbank] [--out-mode combined|per-db|both]\n"
            "  [--metric tanimoto] [--k 200] [--top 50] [--nprobe 64]\n"
            "  [--jobs 1] [--gpu | --no-gpu] [--out results.tsv] [--profile]\n");
        return 1;
    }

    // ── Load shard config ────────────────────────────────────────────────────
    Session sess;
    sess.cfg = cfg;
    try {
        sess.shards.load(cfg.config_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }

    // Override FP params from shard config (authoritative source)
    sess.cfg.fp.radius = sess.shards.fp_radius;
    sess.cfg.fp.nbits  = sess.shards.fp_nbits;

    size_t num_shards = sess.shards.shards.size();

    std::fprintf(stderr,
        "\n=================================================================\n"
        "  search_sharded_database\n"
        "  Config       : %s\n"
        "  Shards       : %zu  |  Total molecules: %s\n"
        "  FP           : radius=%d  nbits=%d\n"
        "  FAISS params : nlist=%d  m=%d  nbits-pq=%d\n"
        "  Per-shard k  : %d    nprobe=%d    jobs=%d\n"
        "=================================================================\n",
        cfg.config_path.c_str(),
        num_shards, fmt_human((size_t)sess.shards.total_molecules).c_str(),
        sess.cfg.fp.radius, sess.cfg.fp.nbits,
        sess.shards.nlist, sess.shards.pq_m, sess.shards.pq_nbits,
        cfg.candidate_k, cfg.nprobe, cfg.jobs);

    // ── GPU selection ─────────────────────────────────────────────────────────
    sess.use_gpu = select_gpu_backend(cfg.force_gpu);
    std::fprintf(stderr, "  Backend      : %s\n", sess.use_gpu ? "GPU" : "CPU");

    // ── Metric ────────────────────────────────────────────────────────────────
    sess.metric = parse_metric(cfg.metric_str);

    // ── Database selection ────────────────────────────────────────────────────
    if (!cfg.databases_str.empty()) {
        parse_databases_flag(cfg.databases_str, sess.use_enamine, sess.use_npass, sess.use_drugbank);
    } else {
        show_database_menu(sess.use_enamine, sess.use_npass, sess.use_drugbank, cfg);
    }
    std::fprintf(stderr, "  Databases    :%s%s%s\n",
                 sess.use_enamine  ? " Enamine"  : "",
                 sess.use_npass    ? " NPASS"    : "",
                 sess.use_drugbank ? " DrugBank" : "");

    // ── Output mode ───────────────────────────────────────────────────────────
    if (!cfg.out_mode_str.empty()) {
        sess.out_mode = parse_out_mode(cfg.out_mode_str);
    } else {
        show_out_mode_menu(sess.out_mode);
    }

    std::fprintf(stderr, "\n");

    // ── Load EnamineReader (global — one reader for all shards) ───────────────
    {
        std::fprintf(stderr, "[INIT] Loading Enamine reader (global offsets)...\n");
        try {
            sess.enamine.open(sess.shards.cxsmiles_path,
                              sess.shards.offsets_path,
                              sess.shards.header_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ERROR] %s\n", e.what());
            return 1;
        }
        std::fprintf(stderr, "  [OK] %s molecules indexed\n\n",
                     fmt_human((size_t)sess.enamine.count()).c_str());
    }

    // ── Load NPASS ────────────────────────────────────────────────────────────
    if (sess.use_npass && !cfg.npass_dir.empty()) {
        std::fprintf(stderr, "[INIT] Loading NPASS...\n");
        try {
            sess.npass.load_from_dir(cfg.npass_dir);
            sess.has_npass = true;
            std::fprintf(stderr, "  [OK] %s compounds\n\n",
                         fmt_human(sess.npass.compound_count()).c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  [WARN] NPASS load failed: %s\n\n", e.what());
        }
    }

    // ── Load DrugBank ─────────────────────────────────────────────────────────
    if (sess.use_drugbank && !cfg.drugbank_bin_path.empty()) {
        std::fprintf(stderr, "[INIT] Loading DrugBank...\n");
        try {
            sess.drugbank.load_binary(cfg.drugbank_bin_path);
            sess.has_drugbank = true;
            std::fprintf(stderr, "  [OK] %zu drugs\n\n", sess.drugbank.drugs().size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  [WARN] DrugBank load failed: %s\n\n", e.what());
        }
    }

    // ── Query loop ────────────────────────────────────────────────────────────
    int query_num = 0;
    std::string first_query = cfg.query_smiles;

    while (true) {
        std::string smiles = first_query;
        first_query.clear();

        if (smiles.empty()) {
            std::fprintf(stderr, "  Enter query SMILES (or 'quit'): ");
            std::fflush(stderr);
            if (!std::getline(std::cin, smiles) || smiles == "quit" || smiles == "q")
                break;
            if (smiles.empty()) continue;
        }

        run_query(sess, smiles, query_num);
        ++query_num;

        std::fprintf(stderr, "  Another search? [Y/n]: ");
        std::fflush(stderr);
        std::string ans;
        if (!std::getline(std::cin, ans)) break;
        if (!ans.empty() && (ans[0]=='n' || ans[0]=='N')) break;
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "\n  Session complete. %d quer%s processed.\n\n",
                 query_num, query_num == 1 ? "y" : "ies");
    return 0;
}
