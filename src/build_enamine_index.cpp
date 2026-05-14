// build_enamine_index: builds FAISS IVFPQ index from Enamine CXSMILES file.
// Requires prepare_enamine to have been run first (produces enamine.offsets.bin).
//
// Usage:
//   build_enamine_index
//     --input   FILE              (Enamine .cxsmiles)
//     --offsets DIR/enamine.offsets.bin
//     --header  DIR/enamine.header.tsv   (optional; guessed if omitted)
//     --out     DIR/faiss.index
//     [--mol-start N]             (first molecule index, 0-based; default 0)
//     [--mol-end N]               (one-past-last molecule index; default = all)
//     [--radius 3]                (Morgan FP radius)
//     [--nbits 4096]              (Morgan FP bit count)
//     [--batch 100000]            (molecules per add-batch)
//     [--nlist 4096]              (FAISS IVF cell count)
//     [--m 32]                    (PQ sub-vectors; nbits/128 recommended)
//     [--nbits-pq 8]              (bits per PQ sub-vector)
//     [--train-size 500000]       (molecules used for FAISS training)
//     [--threads 16]
//     [--gpu]                     (force GPU mode)
//     [--no-gpu]                  (force CPU mode)
//     [--profile]                 (print per-phase timing summary at end)
//
// FAISS IDs stored = global molecule indices (mol_start + local_offset) so a
// single EnamineReader with the full offset file can serve all shards.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include "common/fp_utils.hpp"
#include "common/logger.hpp"
#include "common/progress.hpp"
#include "common/tsv_utils.hpp"
#include "enamine/offset_store.hpp"

namespace fs = std::filesystem;
using namespace bscs;

using Clock = std::chrono::steady_clock;
static double elapsed_s(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// ── CLI ───────────────────────────────────────────────────────────────────────
struct Config {
    std::string input_path;
    std::string offsets_path;
    std::string header_path;
    std::string out_path;
    FpConfig    fp{3, 4096};
    size_t      batch_size  = 100000;
    int         nlist       = 4096;
    int         pq_m        = 32;
    int         pq_nbits    = 8;
    size_t      train_size  = 500000;
    int         threads     = 16;
    bool        profile     = false;
    uint64_t    mol_start   = 0;
    uint64_t    mol_end     = 0;  // 0 = "all"
    int         force_gpu   = -1; // -1=ask, 0=CPU, 1=GPU
};

static Config parse_args(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="--input"      &&i+1<argc) c.input_path   = argv[++i];
        else if (a=="--offsets"    &&i+1<argc) c.offsets_path = argv[++i];
        else if (a=="--header"     &&i+1<argc) c.header_path  = argv[++i];
        else if (a=="--out"        &&i+1<argc) c.out_path      = argv[++i];
        else if (a=="--radius"     &&i+1<argc) c.fp.radius     = std::stoi(argv[++i]);
        else if (a=="--nbits"      &&i+1<argc) c.fp.nbits       = std::stoi(argv[++i]);
        else if (a=="--batch"      &&i+1<argc) c.batch_size    = std::stoull(argv[++i]);
        else if (a=="--nlist"      &&i+1<argc) c.nlist          = std::stoi(argv[++i]);
        else if (a=="--m"          &&i+1<argc) c.pq_m           = std::stoi(argv[++i]);
        else if (a=="--nbits-pq"   &&i+1<argc) c.pq_nbits       = std::stoi(argv[++i]);
        else if (a=="--train-size" &&i+1<argc) c.train_size    = std::stoull(argv[++i]);
        else if (a=="--threads"    &&i+1<argc) c.threads        = std::stoi(argv[++i]);
        else if (a=="--profile")               c.profile        = true;
        else if (a=="--mol-start"  &&i+1<argc) c.mol_start     = std::stoull(argv[++i]);
        else if (a=="--mol-end"    &&i+1<argc) c.mol_end       = std::stoull(argv[++i]);
        else if (a=="--gpu")                   c.force_gpu     = 1;
        else if (a=="--no-gpu")                c.force_gpu     = 0;
    }
    return c;
}

// ── GPU/CPU interactive selection ─────────────────────────────────────────────
// force: -1=ask interactively, 0=CPU, 1=GPU
static bool select_gpu_backend(int force = -1) {
    if (force == 0) { std::fprintf(stderr, "  → CPU mode (forced)\n"); return false; }
    int num_gpus = 0;
    try { num_gpus = faiss::gpu::getNumDevices(); } catch (...) {}
    if (num_gpus == 0) {
        if (force == 1) std::fprintf(stderr, "  [WARN] --gpu requested but no GPU found; using CPU\n");
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

// ── Profile ───────────────────────────────────────────────────────────────────
struct ProfileData {
    double t_smiles_read  = 0;  size_t n_smiles_read = 0;
    double t_fp_train     = 0;  size_t n_fp_train    = 0;
    double t_faiss_train  = 0;
    double t_fp_add       = 0;  // wall-clock across all batches
    double t_idx_add      = 0;  // wall-clock across all batches
    size_t n_batches      = 0;
    double t_transfer     = 0;  // GPU→CPU
    double t_write        = 0;
    double t_total        = 0;
    bool   used_gpu       = false;
};

static void print_profile(const ProfileData& p, const Config& cfg) {
    const char* backend = p.used_gpu ? "GPU" : "CPU";
    double t4 = p.t_fp_add + p.t_idx_add;
    double pct_fp  = (t4 > 0) ? 100.0 * p.t_fp_add  / t4 : 0.0;
    double pct_add = (t4 > 0) ? 100.0 * p.t_idx_add / t4 : 0.0;
    double fp_of_total = (p.t_total > 0) ? 100.0 * p.t_fp_add / p.t_total : 0.0;
    double avg_batch   = (p.n_batches > 0) ? t4 / p.n_batches : 0.0;
    double smiles_rate = (p.t_smiles_read > 0)
                         ? p.n_smiles_read / p.t_smiles_read / 1000.0 : 0;
    double fp_rate     = (p.t_fp_train > 0)
                         ? p.n_fp_train    / p.t_fp_train    / 1000.0 : 0;

    std::fprintf(stderr,
        "\n[PROFILE] Phase 1 - SMILES read:           %7.3fs   (%s mol, %.0fK mol/s)\n",
        p.t_smiles_read, fmt_human(p.n_smiles_read).c_str(), smiles_rate);
    std::fprintf(stderr,
        "[PROFILE] Phase 2 - FP compute (OMP):       %7.3fs   (%s mol, %.0fK mol/s)\n",
        p.t_fp_train, fmt_human(p.n_fp_train).c_str(), fp_rate);
    std::fprintf(stderr,
        "[PROFILE] Phase 3 - FAISS train (%s):      %7.3fs\n",
        backend, p.t_faiss_train);
    std::fprintf(stderr,
        "[PROFILE] Phase 4 - Batch add loop:         %7.2fs   total\n", t4);
    std::fprintf(stderr,
        "[PROFILE]   └─ FP compute (OMP):            %7.2fs   (%4.1f%%)%s\n",
        p.t_fp_add, pct_fp, (p.t_fp_add >= p.t_idx_add) ? "  ← bottleneck" : "");
    std::fprintf(stderr,
        "[PROFILE]   └─ %s add:                      %7.2fs   (%4.1f%%)\n",
        backend, p.t_idx_add, pct_add);
    std::fprintf(stderr,
        "[PROFILE]   └─ avg per batch:               %7.3fs   (%s mol/batch)\n",
        avg_batch, fmt_human(cfg.batch_size).c_str());

    int phase_write = 5;
    if (p.used_gpu) {
        std::fprintf(stderr,
            "[PROFILE] Phase 5 - GPU→CPU transfer:       %7.3fs\n",
            p.t_transfer);
        phase_write = 6;
    }
    std::fprintf(stderr,
        "[PROFILE] Phase %d - Write index:             %7.3fs\n",
        phase_write, p.t_write);
    std::fprintf(stderr,
        "[PROFILE] ──────────────────────────────────────────\n"
        "[PROFILE] Total:                             %7.2fs\n"
        "[PROFILE] Bottleneck: FP compute (%.1f%% of total)\n\n",
        p.t_total, fp_of_total);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.input_path.empty() || cfg.offsets_path.empty() || cfg.out_path.empty()) {
        std::cerr << "Usage: build_enamine_index --input FILE --offsets FILE "
                     "--out FILE [options]\n";
        return 1;
    }
    if (cfg.header_path.empty())
        cfg.header_path = (fs::path(cfg.offsets_path).parent_path()
                           / "enamine.header.tsv").string();

    omp_set_num_threads(cfg.threads);

    bool use_gpu = select_gpu_backend(cfg.force_gpu);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  build_enamine_index\n");
    std::fprintf(stderr, "  Input      : %s\n", cfg.input_path.c_str());
    std::fprintf(stderr, "  Offsets    : %s\n", cfg.offsets_path.c_str());
    std::fprintf(stderr, "  Output     : %s\n", cfg.out_path.c_str());
    std::fprintf(stderr, "  FP         : radius=%d  nbits=%d\n",
                 cfg.fp.radius, cfg.fp.nbits);
    std::fprintf(stderr, "  FAISS IVFPQ: nlist=%d  m=%d  pq_nbits=%d\n",
                 cfg.nlist, cfg.pq_m, cfg.pq_nbits);
    std::fprintf(stderr, "  Batch      : %zu  train_size=%zu  threads=%d\n",
                 cfg.batch_size, cfg.train_size, cfg.threads);
    std::fprintf(stderr, "  Backend    : %s\n",
                 use_gpu ? "GPU (GpuIndexIVFPQ)" : "CPU (IndexIVFPQ)");
    std::fprintf(stderr, "  Profile    : %s\n", cfg.profile ? "ON" : "OFF");
    std::fprintf(stderr, "=================================================================\n");

    OffsetStore offsets(cfg.offsets_path);
    uint64_t total_mols = offsets.count();
    std::fprintf(stderr, "  Molecules in offset file: %s\n",
                 fmt_human(total_mols).c_str());

    if (total_mols == 0) { LOG_ERR("Offset file is empty"); return 1; }

    // Clamp shard bounds
    if (cfg.mol_end == 0 || cfg.mol_end > total_mols) cfg.mol_end = total_mols;
    if (cfg.mol_start >= cfg.mol_end) {
        LOG_ERR("mol_start (%zu) >= mol_end (%zu)", cfg.mol_start, cfg.mol_end);
        return 1;
    }
    uint64_t shard_mols = cfg.mol_end - cfg.mol_start;
    if (cfg.mol_start > 0 || cfg.mol_end < total_mols)
        std::fprintf(stderr, "  Shard range: [%s, %s)  →  %s shard molecules\n",
                     fmt_human(cfg.mol_start).c_str(),
                     fmt_human(cfg.mol_end).c_str(),
                     fmt_human(shard_mols).c_str());
    std::fprintf(stderr, "\n");

    const int DIM = cfg.fp.nbits;
    fs::create_directories(fs::path(cfg.out_path).parent_path());

    std::string err_path = (fs::path(cfg.out_path).parent_path()
                            / "build_errors.smi").string();
    std::ofstream err_out(err_path);
    std::mutex err_mu;

    ProfileData prof;
    prof.used_gpu = use_gpu;
    auto t_total = Clock::now();
    char buf[65536];

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: read training SMILES from offset-indexed positions
    // ─────────────────────────────────────────────────────────────────────────
    size_t actual_train = std::min(cfg.train_size, (size_t)shard_mols);

    {
        auto t1 = Clock::now();
        Progress p1("Phase 1: reading train SMILES", actual_train, "mol");

        FILE* ft = std::fopen(cfg.input_path.c_str(), "rb");
        if (!ft) { LOG_ERR("Cannot open input"); return 1; }

        std::vector<std::string> train_smiles;
        train_smiles.reserve(actual_train);

        for (uint64_t i = 0; i < actual_train; i++) {
            std::fseek(ft, (long)offsets[cfg.mol_start + i], SEEK_SET);
            if (!std::fgets(buf, sizeof(buf), ft)) break;
            std::string_view sv(buf, std::strlen(buf));
            while (!sv.empty() && (sv.back()=='\n'||sv.back()=='\r')) sv.remove_suffix(1);
            train_smiles.push_back(strip_cxsmiles(first_field(sv)));
            p1.update(i + 1);
        }
        std::fclose(ft);
        p1.finish(train_smiles.size());
        prof.t_smiles_read = elapsed_s(t1);
        prof.n_smiles_read = train_smiles.size();

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: compute Morgan fingerprints for training set (parallel)
        // ─────────────────────────────────────────────────────────────────────
        std::vector<float> train_mat((size_t)actual_train * DIM, 0.0f);
        std::atomic<size_t> valid_train{0};
        std::atomic<size_t> fp_done{0};

        {
            auto t2 = Clock::now();
            Progress p2("Phase 2: computing train fingerprints", actual_train, "mol");

            #pragma omp parallel for schedule(dynamic, 256)
            for (long i = 0; i < (long)train_smiles.size(); i++) {
                std::vector<float> tmp(DIM);
                bool ok = smiles_to_fp_float(train_smiles[i], cfg.fp, tmp.data());
                if (!ok) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    err_out << train_smiles[i] << "\n";
                } else {
                    size_t slot = valid_train.fetch_add(1);
                    std::copy(tmp.begin(), tmp.end(), train_mat.data() + slot * DIM);
                }
                size_t done = fp_done.fetch_add(1) + 1;
                if (omp_get_thread_num() == 0 && (done & 0x3FFF) == 0)
                    p2.update(done);
            }
            p2.finish(fp_done.load());
            prof.t_fp_train = elapsed_s(t2);
            prof.n_fp_train = fp_done.load();
        }

        train_smiles.clear();
        train_smiles.shrink_to_fit();

        std::fprintf(stderr, "  Valid training vectors: %s / %s\n\n",
                     fmt_human(valid_train.load()).c_str(),
                     fmt_human(actual_train).c_str());

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: FAISS index training (GPU or CPU)
        // ─────────────────────────────────────────────────────────────────────
        faiss::IndexFlatL2  cpu_quantizer(DIM);
        faiss::IndexIVFPQ   cpu_index(&cpu_quantizer, DIM, cfg.nlist, cfg.pq_m, cfg.pq_nbits);

        std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;
        std::unique_ptr<faiss::gpu::GpuIndexIVFPQ>        gpu_index;

        {
            char extra[160];
            std::snprintf(extra, sizeof(extra),
                "nlist=%d  m=%d  vectors=%s  backend=%s",
                cfg.nlist, cfg.pq_m, fmt_human(valid_train.load()).c_str(),
                use_gpu ? "GPU" : "CPU");
            Spinner s3("Phase 3: FAISS IVFPQ training");
            s3.tick(extra);

            auto t3 = Clock::now();
            if (use_gpu) {
                gpu_res   = std::make_unique<faiss::gpu::StandardGpuResources>();
                gpu_index = std::make_unique<faiss::gpu::GpuIndexIVFPQ>(
                    gpu_res.get(), DIM,
                    (faiss::idx_t)cfg.nlist,
                    (faiss::idx_t)cfg.pq_m,
                    (faiss::idx_t)cfg.pq_nbits);
                gpu_index->train((faiss::idx_t)valid_train.load(), train_mat.data());
            } else {
                cpu_index.train((faiss::idx_t)valid_train.load(), train_mat.data());
            }
            prof.t_faiss_train = elapsed_s(t3);
            s3.finish(extra);
        }

        train_mat.clear();
        train_mat.shrink_to_fit();

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4: stream all molecules, compute FP, add to FAISS with IDs
        //   t_fp_add  = wall-clock time for OMP FP compute across all batches
        //   t_idx_add = wall-clock time for index.add_with_ids across all batches
        // ─────────────────────────────────────────────────────────────────────
        Progress p4("Phase 4: indexing Enamine molecules", shard_mols, "mol");

        FILE* fin = std::fopen(cfg.input_path.c_str(), "rb");
        if (!fin) { LOG_ERR("Cannot reopen input"); return 1; }

        size_t batch_size = cfg.batch_size;
        std::vector<float>        fp_batch(batch_size * DIM);
        std::vector<std::string>  smi_batch(batch_size);
        std::vector<uint64_t>     id_batch(batch_size);
        std::vector<char>         valid_batch(batch_size, 0);
        std::vector<faiss::idx_t> faiss_ids(batch_size);

        uint64_t mol_cursor    = 0;   // local cursor within [0, shard_mols)
        uint64_t total_added   = 0;
        uint64_t total_skipped = 0;

        while (mol_cursor < shard_mols) {
            size_t bsz = std::min(batch_size, (size_t)(shard_mols - mol_cursor));

            for (size_t i = 0; i < bsz; i++) {
                // Global offset: mol_start + local position
                std::fseek(fin, (long)offsets[cfg.mol_start + mol_cursor + i], SEEK_SET);
                if (!std::fgets(buf, sizeof(buf), fin)) { smi_batch[i].clear(); continue; }
                std::string_view sv(buf, std::strlen(buf));
                while (!sv.empty() && (sv.back()=='\n'||sv.back()=='\r')) sv.remove_suffix(1);
                smi_batch[i] = strip_cxsmiles(first_field(sv));
                id_batch[i]  = cfg.mol_start + mol_cursor + i;  // global FAISS ID
            }

            std::fill(valid_batch.begin(), valid_batch.begin() + bsz, 0);

            // Timed OMP FP compute
            auto t_fp0 = Clock::now();
            #pragma omp parallel for schedule(dynamic, 256)
            for (long i = 0; i < (long)bsz; i++) {
                if (smi_batch[i].empty()) continue;
                bool ok = smiles_to_fp_float(smi_batch[i], cfg.fp,
                                             fp_batch.data() + i * DIM);
                if (ok) valid_batch[i] = 1;
                else {
                    std::lock_guard<std::mutex> lk(err_mu);
                    err_out << smi_batch[i] << "\n";
                }
            }
            prof.t_fp_add += elapsed_s(t_fp0);

            // Compact valid entries, preserving original FAISS IDs
            size_t ptr = 0;
            for (size_t i = 0; i < bsz; i++) {
                if (!valid_batch[i]) { ++total_skipped; continue; }
                if (ptr != i)
                    std::copy(fp_batch.data() + i * DIM,
                              fp_batch.data() + (i+1) * DIM,
                              fp_batch.data() + ptr * DIM);
                faiss_ids[ptr] = (faiss::idx_t)id_batch[i];
                ++ptr;
            }

            // Timed index add
            auto t_add0 = Clock::now();
            if (ptr > 0) {
                if (use_gpu)
                    gpu_index->add_with_ids((faiss::idx_t)ptr, fp_batch.data(), faiss_ids.data());
                else
                    cpu_index.add_with_ids((faiss::idx_t)ptr, fp_batch.data(), faiss_ids.data());
            }
            prof.t_idx_add += elapsed_s(t_add0);

            total_added  += ptr;
            mol_cursor   += bsz;
            ++prof.n_batches;
            p4.update(mol_cursor);
        }
        std::fclose(fin);
        err_out.flush();
        p4.finish(mol_cursor);

        std::fprintf(stderr, "\n  Added  : %s\n", fmt_human(total_added).c_str());
        std::fprintf(stderr, "  Skipped: %s (invalid SMILES)\n\n",
                     fmt_human(total_skipped).c_str());

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4.5 (GPU only): transfer GPU index → CPU for serialization
        // ─────────────────────────────────────────────────────────────────────
        if (use_gpu) {
            Spinner s45("Phase 4.5: GPU → CPU transfer");
            s45.tick("copying inverted lists from device");
            auto t45 = Clock::now();
            gpu_index->copyTo(&cpu_index);
            prof.t_transfer = elapsed_s(t45);
            gpu_index.reset();
            gpu_res.reset();
            s45.finish("done");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 5 / 6: write FAISS index to disk
        // ─────────────────────────────────────────────────────────────────────
        {
            const char* label = use_gpu ? "Phase 6: writing FAISS index"
                                        : "Phase 5: writing FAISS index";
            Spinner s_write(label);
            s_write.tick(cfg.out_path.c_str());
            auto t_w = Clock::now();
            faiss::write_index(&cpu_index, cfg.out_path.c_str());
            prof.t_write = elapsed_s(t_w);
            s_write.finish(cfg.out_path.c_str());
        }
    }

    prof.t_total = elapsed_s(t_total);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  build_enamine_index complete.\n");
    std::fprintf(stderr, "  Total time : %s\n", fmt_elapsed(prof.t_total).c_str());
    std::fprintf(stderr, "  Index      : %s\n", cfg.out_path.c_str());
    std::fprintf(stderr, "  Errors     : %s\n", err_path.c_str());
    std::fprintf(stderr, "=================================================================\n\n");

    if (cfg.profile) print_profile(prof, cfg);

    return 0;
}
