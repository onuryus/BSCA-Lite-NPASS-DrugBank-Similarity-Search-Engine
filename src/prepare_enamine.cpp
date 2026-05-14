// prepare_enamine: scans Enamine CXSMILES file, records byte offsets per
// molecule, writes offset index + header + schema.
//
// Usage:
//   prepare_enamine --input FILE --outdir DIR

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "common/logger.hpp"
#include "common/progress.hpp"
#include "common/tsv_utils.hpp"
#include "enamine/offset_store.hpp"

namespace fs = std::filesystem;
using namespace bscs;

static void write_schema_json(const std::string& path,
                               const std::vector<std::string>& cols) {
    std::ofstream f(path);
    f << "{\n  \"columns\": [\n";
    for (size_t i = 0; i < cols.size(); ++i)
        f << "    " << std::quoted(cols[i])
          << (i + 1 < cols.size() ? "," : "") << "\n";
    f << "  ]\n}\n";
}

static uint64_t file_size_bytes(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) return (uint64_t)st.st_size;
    return 0;
}

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string outdir;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--input"  && i+1 < argc) input_path = argv[++i];
        else if (a == "--outdir" && i+1 < argc) outdir     = argv[++i];
    }

    if (input_path.empty() || outdir.empty()) {
        std::cerr << "Usage: prepare_enamine --input FILE --outdir DIR\n";
        return 1;
    }

    fs::create_directories(outdir);

    const std::string offsets_path = (fs::path(outdir) / "enamine.offsets.bin").string();
    const std::string header_path  = (fs::path(outdir) / "enamine.header.tsv").string();
    const std::string schema_path  = (fs::path(outdir) / "enamine.schema.json").string();
    const std::string count_path   = (fs::path(outdir) / "enamine.count").string();

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_enamine\n");
    std::fprintf(stderr, "  Input : %s\n", input_path.c_str());
    std::fprintf(stderr, "  Outdir: %s\n", outdir.c_str());
    std::fprintf(stderr, "=================================================================\n");

    // ── get file size for progress denominator ────────────────────────────────
    uint64_t total_bytes = file_size_bytes(input_path);
    std::fprintf(stderr, "  File size: %s\n\n", fmt_bytes(total_bytes).c_str());

    FILE* fin = std::fopen(input_path.c_str(), "rb");
    if (!fin) { LOG_ERR("Cannot open input file"); return 1; }

    OffsetWriter ow(offsets_path);

    // ── Phase 1: read and write header ────────────────────────────────────────
    std::fprintf(stderr, "[PHASE] Phase 1: reading header\n");
    char buf[65536];
    if (!std::fgets(buf, sizeof(buf), fin)) {
        std::fclose(fin); LOG_ERR("Empty input file"); return 1;
    }
    std::string header_line(buf);
    while (!header_line.empty() &&
           (header_line.back() == '\n' || header_line.back() == '\r'))
        header_line.pop_back();

    auto cols = parse_header(header_line);
    std::fprintf(stderr, "  [OK]  Found %zu columns\n", cols.size());
    for (size_t i = 0; i < cols.size(); ++i)
        std::fprintf(stderr, "        [%2zu] %s\n", i, cols[i].c_str());

    { std::ofstream hf(header_path); hf << header_line << "\n"; }
    write_schema_json(schema_path, cols);
    std::fprintf(stderr, "  [OK]  Header written: %s\n", header_path.c_str());
    std::fprintf(stderr, "  [OK]  Schema written : %s\n\n", schema_path.c_str());

    // ── Phase 2: scan molecules, record offsets ───────────────────────────────
    // Use file position as progress denominator (byte-level, no pre-count needed)
    ProgressBytes prog("Phase 2: scanning Enamine offsets",
                       total_bytes > 0 ? total_bytes : 1);

    uint64_t mol_count = 0;

    while (true) {
        long pos = (long)std::ftell(fin);
        if (!std::fgets(buf, sizeof(buf), fin)) break;
        size_t len = std::strlen(buf);
        if (len == 0 || buf[0] == '\n' || buf[0] == '\r') continue;

        // Consume remainder of lines longer than the buffer
        while (len == sizeof(buf) - 1 && buf[len-1] != '\n') {
            if (!std::fgets(buf, sizeof(buf), fin)) break;
            len = std::strlen(buf);
        }

        ow.write(static_cast<uint64_t>(pos));
        ++mol_count;

        if ((mol_count & 0x1FFFF) == 0) {  // every ~131K molecules
            uint64_t cur_pos = (uint64_t)std::ftell(fin);
            prog.update(cur_pos, mol_count, "mol");
        }
    }
    ow.flush();

    {
        uint64_t final_pos = total_bytes > 0 ? total_bytes : (uint64_t)std::ftell(fin);
        prog.finish(final_pos, mol_count, "mol");
    }
    std::fclose(fin);

    // ── Phase 3: write count file ─────────────────────────────────────────────
    std::fprintf(stderr, "\n[PHASE] Phase 3: writing metadata\n");
    { std::ofstream cf(count_path); cf << mol_count << "\n"; }
    std::fprintf(stderr, "  [OK]  Molecule count : %s\n",
                 fmt_human(mol_count).c_str());
    std::fprintf(stderr, "  [OK]  Offsets written: %s\n", offsets_path.c_str());
    std::fprintf(stderr, "  [OK]  Count written  : %s\n\n", count_path.c_str());

    std::fprintf(stderr, "=================================================================\n");
    std::fprintf(stderr, "  prepare_enamine complete.\n");
    std::fprintf(stderr, "  Total molecules : %s\n", fmt_human(mol_count).c_str());
    std::fprintf(stderr, "  Offset file     : %s\n", offsets_path.c_str());
    std::fprintf(stderr, "=================================================================\n\n");

    return 0;
}
