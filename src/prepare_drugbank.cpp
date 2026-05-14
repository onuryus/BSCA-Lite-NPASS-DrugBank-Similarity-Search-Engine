// prepare_drugbank: parses DrugBank XML and writes a fast binary lookup index.
//
// Usage:
//   prepare_drugbank --xml FILE --outdir DIR
//
// Outputs:
//   DIR/drugbank.bin   — binary drug records (fast load for search_annotate)

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include "common/logger.hpp"
#include "common/progress.hpp"
#include "drugbank/drugbank_index.hpp"

namespace fs = std::filesystem;
using namespace bscs;

static uint64_t file_size_bytes(const std::string& path) {
    struct stat st{};
    return (::stat(path.c_str(), &st) == 0) ? (uint64_t)st.st_size : 0;
}

int main(int argc, char* argv[]) {
    std::string xml_path, outdir;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--xml"    && i+1 < argc) xml_path = argv[++i];
        else if (a == "--outdir" && i+1 < argc) outdir   = argv[++i];
    }

    if (xml_path.empty() || outdir.empty()) {
        std::cerr << "Usage: prepare_drugbank --xml FILE --outdir DIR\n";
        return 1;
    }

    fs::create_directories(outdir);

    const std::string bin_path = (fs::path(outdir) / "drugbank.bin").string();

    uint64_t total_bytes = file_size_bytes(xml_path);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_drugbank\n");
    std::fprintf(stderr, "  Input  : %s\n", xml_path.c_str());
    std::fprintf(stderr, "  Outdir : %s\n", outdir.c_str());
    std::fprintf(stderr, "  Size   : %s\n", fmt_bytes(total_bytes).c_str());
    std::fprintf(stderr, "=================================================================\n");

    auto t_total = std::chrono::steady_clock::now();

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: parse DrugBank XML
    // ─────────────────────────────────────────────────────────────────────────
    ProgressBytes p1("Phase 1: parsing DrugBank XML", total_bytes > 0 ? total_bytes : 1);

    DrugBankIndex index;
    index.load_from_xml(xml_path,
        [&](uint64_t bytes_done, uint64_t /*total*/, size_t drugs_done) {
            p1.update(bytes_done, drugs_done, "drugs");
        });

    p1.finish(total_bytes, index.drug_count(), "drugs");

    size_t total_drugs   = index.drug_count();
    size_t with_inchikey = 0;
    size_t with_smiles   = 0;
    size_t with_targets  = 0;
    for (const auto& d : index.drugs()) {
        if (!d.inchikey.empty()) ++with_inchikey;
        if (!d.smiles.empty())   ++with_smiles;
        if (!d.targets.empty())  ++with_targets;
    }

    std::fprintf(stderr, "\n  Total drugs   : %s\n", fmt_human(total_drugs).c_str());
    std::fprintf(stderr, "  With InChIKey : %s (%.1f%%)\n",
                 fmt_human(with_inchikey).c_str(),
                 total_drugs ? 100.0 * with_inchikey / total_drugs : 0.0);
    std::fprintf(stderr, "  With SMILES   : %s (%.1f%%)\n",
                 fmt_human(with_smiles).c_str(),
                 total_drugs ? 100.0 * with_smiles / total_drugs : 0.0);
    std::fprintf(stderr, "  With targets  : %s (%.1f%%)\n\n",
                 fmt_human(with_targets).c_str(),
                 total_drugs ? 100.0 * with_targets / total_drugs : 0.0);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: write binary index
    // ─────────────────────────────────────────────────────────────────────────
    {
        Spinner s2("Phase 2: writing binary index");
        s2.tick(bin_path.c_str());
        index.save_binary(bin_path);
        s2.finish(bin_path.c_str());
    }

    uint64_t bin_bytes = file_size_bytes(bin_path);
    std::fprintf(stderr, "  [OK]  %s  (%s)\n\n", bin_path.c_str(),
                 fmt_bytes(bin_bytes).c_str());

    double total_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_total).count();

    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_drugbank complete.\n");
    std::fprintf(stderr, "  Drugs   : %s\n",      fmt_human(total_drugs).c_str());
    std::fprintf(stderr, "  Index   : %s\n",      bin_path.c_str());
    std::fprintf(stderr, "  Time    : %s\n",      fmt_elapsed(total_sec).c_str());
    std::fprintf(stderr, "=================================================================\n\n");

    return 0;
}
