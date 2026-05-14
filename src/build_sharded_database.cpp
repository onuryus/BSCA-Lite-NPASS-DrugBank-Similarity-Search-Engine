// build_sharded_database — coordinator tool that builds a sharded FAISS index
// database from a single large Enamine CXSMILES file.
//
// Workflow:
//   1. Run prepare_enamine once to build the global offset file.
//   2. Split the molecule range into N equal shards.
//   3. Call build_enamine_index for each shard (mol_start/mol_end + global IDs).
//   4. Write shard_config.json so search_sharded_database can find all shards.
//
// Usage:
//   build_sharded_database
//     --input   enamine.cxsmiles
//     --outdir  /data/index
//     [--num-shards  N   | --mols-per-shard M]   (one required)
//     [--skip-prepare]      (skip prepare_enamine if offsets already exist)
//     [--gpu | --no-gpu]
//     [--radius 3] [--nbits 1024]
//     [--nlist 4096] [--m 16] [--nbits-pq 8]
//     [--train-size 1000000] [--batch 1000000]
//     [--threads 12]
//     [--yes]               (skip confirmation prompt)

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/shard_config.hpp"
#include "enamine/offset_store.hpp"

namespace fs = std::filesystem;
using namespace bscs;

// ── RAM estimation ─────────────────────────────────────────────────────────────
static double ram_per_shard_gb(uint64_t n_mols, int pq_m) {
    double pq_ids    = (double)n_mols * (pq_m + 8);          // PQ codes + IDs
    double offsets   = (double)n_mols * 8;                   // offset table
    return (pq_ids + offsets) / (1024.0 * 1024.0 * 1024.0);
}

static std::string fmt_gb(double gb) {
    char buf[32];
    if (gb >= 1000.0) std::snprintf(buf, sizeof(buf), "%.1f TB", gb / 1024.0);
    else              std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
    return buf;
}

static std::string fmt_human(uint64_t n) {
    char buf[32];
    if      (n >= 1000000000ULL) std::snprintf(buf, sizeof(buf), "%.2fB", n / 1.0e9);
    else if (n >= 1000000ULL)    std::snprintf(buf, sizeof(buf), "%.2fM", n / 1.0e6);
    else if (n >= 1000ULL)       std::snprintf(buf, sizeof(buf), "%.1fK", n / 1.0e3);
    else                         std::snprintf(buf, sizeof(buf), "%zu", (size_t)n);
    return buf;
}

// ── CLI ────────────────────────────────────────────────────────────────────────
struct Config {
    std::string input_path;
    std::string outdir;
    uint64_t    num_shards      = 0;
    uint64_t    mols_per_shard  = 0;  // alternative to num_shards
    bool        skip_prepare    = false;
    int         force_gpu       = -1; // -1=ask, 0=CPU, 1=GPU
    int         fp_radius       = 3;
    int         fp_nbits        = 1024;
    int         nlist           = 4096;
    int         pq_m            = 16;
    int         pq_nbits        = 8;
    size_t      train_size      = 1000000;
    size_t      batch_size      = 1000000;
    int         threads         = 12;
    bool        yes             = false;  // skip confirmation
};

static Config parse_args(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="--input"          &&i+1<argc) c.input_path      = argv[++i];
        else if (a=="--outdir"         &&i+1<argc) c.outdir          = argv[++i];
        else if (a=="--num-shards"     &&i+1<argc) c.num_shards      = std::stoull(argv[++i]);
        else if (a=="--mols-per-shard" &&i+1<argc) c.mols_per_shard  = std::stoull(argv[++i]);
        else if (a=="--skip-prepare")              c.skip_prepare    = true;
        else if (a=="--gpu")                       c.force_gpu       = 1;
        else if (a=="--no-gpu")                    c.force_gpu       = 0;
        else if (a=="--radius"         &&i+1<argc) c.fp_radius       = std::stoi(argv[++i]);
        else if (a=="--nbits"          &&i+1<argc) c.fp_nbits        = std::stoi(argv[++i]);
        else if (a=="--nlist"          &&i+1<argc) c.nlist           = std::stoi(argv[++i]);
        else if (a=="--m"              &&i+1<argc) c.pq_m            = std::stoi(argv[++i]);
        else if (a=="--nbits-pq"       &&i+1<argc) c.pq_nbits        = std::stoi(argv[++i]);
        else if (a=="--train-size"     &&i+1<argc) c.train_size      = std::stoull(argv[++i]);
        else if (a=="--batch"          &&i+1<argc) c.batch_size      = std::stoull(argv[++i]);
        else if (a=="--threads"        &&i+1<argc) c.threads         = std::stoi(argv[++i]);
        else if (a=="--yes")                       c.yes             = true;
    }
    return c;
}

// ── Tool path ─────────────────────────────────────────────────────────────────
static std::string tool_dir(const char* argv0) {
    char rp[PATH_MAX] = {};
    if (realpath(argv0, rp)) return fs::path(rp).parent_path().string();
    return ".";
}

// ── Shell command (logged + executed) ─────────────────────────────────────────
static int run_cmd(const std::string& cmd) {
    std::fprintf(stderr, "\n[CMD] %s\n\n", cmd.c_str());
    return std::system(cmd.c_str());
}

// ── Quote a path for shell insertion ─────────────────────────────────────────
static std::string shq(const std::string& s) { return "\"" + s + "\""; }

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.input_path.empty() || cfg.outdir.empty()) {
        std::fprintf(stderr,
            "Usage: build_sharded_database\n"
            "  --input   enamine.cxsmiles\n"
            "  --outdir  /data/index\n"
            "  [--num-shards N | --mols-per-shard M]\n"
            "  [--skip-prepare] [--gpu | --no-gpu] [--yes]\n"
            "  [--radius 3] [--nbits 1024] [--nlist 4096]\n"
            "  [--m 16] [--nbits-pq 8] [--train-size 1000000]\n"
            "  [--batch 1000000] [--threads 12]\n");
        return 1;
    }
    if (cfg.num_shards == 0 && cfg.mols_per_shard == 0) {
        std::fprintf(stderr,
            "[ERROR] Specify --num-shards N or --mols-per-shard M\n");
        return 1;
    }

    const std::string tdir        = tool_dir(argv[0]);
    const std::string offsets_path = cfg.outdir + "/enamine.offsets.bin";
    const std::string header_path  = cfg.outdir + "/enamine.header.tsv";
    const std::string config_path  = cfg.outdir + "/shard_config.json";

    fs::create_directories(cfg.outdir);

    std::fprintf(stderr,
        "\n=================================================================\n"
        "  build_sharded_database\n"
        "  Input   : %s\n"
        "  Outdir  : %s\n"
        "  Config  : %s\n"
        "=================================================================\n\n",
        cfg.input_path.c_str(), cfg.outdir.c_str(), config_path.c_str());

    // ─────────────────────────────────────────────────────────────────────────
    // Step 1: prepare_enamine (build global offset file)
    // ─────────────────────────────────────────────────────────────────────────
    if (!cfg.skip_prepare || !fs::exists(offsets_path)) {
        std::fprintf(stderr, "[STEP 1] Running prepare_enamine ...\n");
        // prepare_enamine writes enamine.offsets.bin + enamine.header.tsv into --outdir
        std::string cmd = tdir + "/prepare_enamine"
            " --input "  + shq(cfg.input_path)
            + " --outdir " + shq(cfg.outdir);
        int rc = run_cmd(cmd);
        if (rc != 0) {
            std::fprintf(stderr, "[ERROR] prepare_enamine failed (rc=%d)\n", rc);
            return 1;
        }
    } else {
        std::fprintf(stderr, "[STEP 1] Skipping prepare_enamine (offsets exist).\n\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 2: count molecules
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[STEP 2] Reading molecule count from offset file ...\n");
    uint64_t total_mols = 0;
    {
        try {
            OffsetStore store(offsets_path);
            total_mols = store.count();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ERROR] %s\n", e.what());
            return 1;
        }
    }
    std::fprintf(stderr, "  Total molecules: %s\n\n", fmt_human(total_mols).c_str());
    if (total_mols == 0) { std::fprintf(stderr, "[ERROR] Offset file is empty.\n"); return 1; }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 3: compute shard boundaries
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr, "[STEP 3] Computing shard layout ...\n");

    if (cfg.mols_per_shard > 0 && cfg.num_shards == 0) {
        cfg.num_shards = (total_mols + cfg.mols_per_shard - 1) / cfg.mols_per_shard;
    }
    if (cfg.num_shards == 0) cfg.num_shards = 1;

    std::vector<ShardEntry> shards;
    {
        uint64_t base = total_mols / cfg.num_shards;
        uint64_t rem  = total_mols % cfg.num_shards;
        uint64_t cur  = 0;
        for (uint64_t s = 0; s < cfg.num_shards; s++) {
            ShardEntry e;
            e.id        = (int)s;
            e.mol_start = cur;
            e.mol_end   = cur + base + (s < rem ? 1 : 0);
            cur         = e.mol_end;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "shard_%03llu", (unsigned long long)s + 1);
            e.faiss_path = cfg.outdir + "/" + buf + "/faiss.index";
            shards.push_back(e);
        }
    }

    uint64_t avg_shard = (total_mols + cfg.num_shards - 1) / cfg.num_shards;
    double   ram_each  = ram_per_shard_gb(avg_shard, cfg.pq_m);
    double   disk_each = (double)avg_shard * (cfg.pq_m + 8 + 8) / (1024.0 * 1024.0 * 1024.0);

    std::fprintf(stderr,
        "  Shards          : %zu\n"
        "  Avg shard size  : %s molecules\n"
        "  RAM / shard     : %s  (m=%d)\n"
        "  Disk / shard    : %s  (FAISS index)\n"
        "  Total disk      : %s  (indices only)\n\n",
        shards.size(),
        fmt_human(avg_shard).c_str(),
        fmt_gb(ram_each).c_str(), cfg.pq_m,
        fmt_gb(disk_each).c_str(),
        fmt_gb(disk_each * shards.size()).c_str());

    std::fprintf(stderr,
        "  FAISS params    : radius=%d nbits=%d nlist=%d m=%d nbits-pq=%d\n"
        "  Threads         : %d   batch=%s   train=%s\n\n",
        cfg.fp_radius, cfg.fp_nbits, cfg.nlist, cfg.pq_m, cfg.pq_nbits,
        cfg.threads,
        fmt_human(cfg.batch_size).c_str(),
        fmt_human(cfg.train_size).c_str());

    if (!cfg.yes) {
        std::fprintf(stderr, "  Build %zu shard(s)? [Y/n]: ", shards.size());
        std::fflush(stderr);
        std::string ans;
        std::getline(std::cin, ans);
        if (!ans.empty() && (ans[0]=='n' || ans[0]=='N')) {
            std::fprintf(stderr, "  Aborted.\n");
            return 0;
        }
        std::fprintf(stderr, "\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 3.5: resolve GPU choice once — propagate to every shard subprocess
    // ─────────────────────────────────────────────────────────────────────────
    bool use_gpu_build = false;
    if (cfg.force_gpu == 1) {
        use_gpu_build = true;
        std::fprintf(stderr, "  GPU: forced ON  (--gpu)\n\n");
    } else if (cfg.force_gpu == 0) {
        use_gpu_build = false;
        std::fprintf(stderr, "  GPU: forced OFF  (--no-gpu, CPU build)\n\n");
    } else {
        // Ask once here; the answer is passed as --gpu / --no-gpu to every shard
        // so build_enamine_index never prompts again.
        std::fprintf(stderr, "  Use GPU for FAISS training? [Y/n]: ");
        std::fflush(stderr);
        std::string ans;
        std::getline(std::cin, ans);
        use_gpu_build = (ans.empty() || ans[0] == 'y' || ans[0] == 'Y');
        std::fprintf(stderr, "  → %s mode selected for all shards\n\n",
                     use_gpu_build ? "GPU" : "CPU");
    }
    const std::string gpu_flag = use_gpu_build ? " --gpu" : " --no-gpu";

    // ─────────────────────────────────────────────────────────────────────────
    // Step 4: build each shard
    // ─────────────────────────────────────────────────────────────────────────

    for (size_t s = 0; s < shards.size(); s++) {
        const auto& sh = shards[s];
        std::fprintf(stderr,
            "\n=================================================================\n"
            "[STEP 4.%zu/%zu] Building shard %03d  [mol %s..%s)\n"
            "=================================================================\n",
            s + 1, shards.size(), sh.id + 1,
            fmt_human(sh.mol_start).c_str(),
            fmt_human(sh.mol_end).c_str());

        fs::create_directories(fs::path(sh.faiss_path).parent_path());

        // Skip if already built
        if (fs::exists(sh.faiss_path)) {
            std::fprintf(stderr, "  [SKIP] faiss.index exists, skipping shard.\n");
            continue;
        }

        std::string cmd = tdir + "/build_enamine_index"
            + " --input "     + shq(cfg.input_path)
            + " --offsets "   + shq(offsets_path)
            + " --header "    + shq(header_path)
            + " --out "       + shq(sh.faiss_path)
            + " --mol-start " + std::to_string(sh.mol_start)
            + " --mol-end "   + std::to_string(sh.mol_end)
            + " --radius "    + std::to_string(cfg.fp_radius)
            + " --nbits "     + std::to_string(cfg.fp_nbits)
            + " --nlist "     + std::to_string(cfg.nlist)
            + " --m "         + std::to_string(cfg.pq_m)
            + " --nbits-pq "  + std::to_string(cfg.pq_nbits)
            + " --train-size "+ std::to_string(cfg.train_size)
            + " --batch "     + std::to_string(cfg.batch_size)
            + " --threads "   + std::to_string(cfg.threads)
            + gpu_flag;

        int rc = run_cmd(cmd);
        if (rc != 0) {
            std::fprintf(stderr,
                "\n[ERROR] Shard %d failed (rc=%d). "
                "Fix the error and re-run — built shards are not re-built.\n",
                sh.id + 1, rc);
            return 1;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Step 5: write shard_config.json
    // ─────────────────────────────────────────────────────────────────────────
    std::fprintf(stderr,
        "\n=================================================================\n"
        "[STEP 5] Writing %s\n"
        "=================================================================\n\n",
        config_path.c_str());

    ShardConfig sc;
    sc.total_molecules = total_mols;
    sc.cxsmiles_path   = fs::absolute(cfg.input_path).string();
    sc.offsets_path    = fs::absolute(offsets_path).string();
    sc.header_path     = fs::absolute(header_path).string();
    sc.fp_radius       = cfg.fp_radius;
    sc.fp_nbits        = cfg.fp_nbits;
    sc.nlist           = cfg.nlist;
    sc.pq_m            = cfg.pq_m;
    sc.pq_nbits        = cfg.pq_nbits;
    for (auto& sh : shards) {
        ShardEntry e = sh;
        e.faiss_path = fs::absolute(sh.faiss_path).string();
        sc.shards.push_back(e);
    }
    sc.save(config_path);

    std::fprintf(stderr,
        "  shard_config.json written.\n"
        "  %zu shards  |  %s total molecules\n\n"
        "  Run searches with:\n"
        "    ./search_sharded_database --config %s [options]\n\n",
        sc.shards.size(),
        fmt_human(sc.total_molecules).c_str(),
        config_path.c_str());

    return 0;
}
