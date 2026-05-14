#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace bscs {

// ── number/time formatters ─────────────────────────────────────────────────────

inline std::string fmt_human(uint64_t n) {
    char buf[32];
    if      (n >= 1'000'000'000ULL) std::snprintf(buf,sizeof(buf),"%.2fB", n/1.0e9);
    else if (n >= 1'000'000ULL)     std::snprintf(buf,sizeof(buf),"%.2fM", n/1.0e6);
    else if (n >= 1'000ULL)         std::snprintf(buf,sizeof(buf),"%.2fK", n/1.0e3);
    else                            std::snprintf(buf,sizeof(buf),"%llu",  (unsigned long long)n);
    return buf;
}

inline std::string fmt_bytes(uint64_t b) {
    char buf[32];
    if      (b >= 1ULL<<30) std::snprintf(buf,sizeof(buf),"%.2f GB", b/(double)(1ULL<<30));
    else if (b >= 1ULL<<20) std::snprintf(buf,sizeof(buf),"%.2f MB", b/(double)(1ULL<<20));
    else if (b >= 1ULL<<10) std::snprintf(buf,sizeof(buf),"%.2f KB", b/(double)(1ULL<<10));
    else                    std::snprintf(buf,sizeof(buf),"%llu B",   (unsigned long long)b);
    return buf;
}

inline std::string fmt_eta(double s) {
    if (s < 0 || s > 86400.0*30) return "--:--:--";
    char buf[32];
    int h  = (int)(s / 3600);
    int m  = (int)((s - h*3600) / 60);
    int sc = (int)s % 60;
    if (h > 0) std::snprintf(buf,sizeof(buf),"%dh %02dm %02ds", h, m, sc);
    else        std::snprintf(buf,sizeof(buf),"%dm %02ds", m, sc);
    return buf;
}

inline std::string fmt_elapsed(double s) {
    char buf[32];
    int h  = (int)(s / 3600);
    int m  = (int)((s - h*3600) / 60);
    int sc = (int)s % 60;
    if      (h > 0) std::snprintf(buf,sizeof(buf),"%dh%02dm%02ds", h, m, sc);
    else if (m > 0) std::snprintf(buf,sizeof(buf),"%dm%02ds", m, sc);
    else            std::snprintf(buf,sizeof(buf),"%.1fs", s);
    return buf;
}

// ── Progress: in-place \r progress bar (known total) ─────────────────────────
//
// Usage:
//   Progress p("Phase 2: indexing", total_mols, "mol");
//   for (uint64_t i = 0; i < total_mols; i++) {
//       do_work();
//       p.update(i + 1);    // throttled internally to ~10 Hz
//   }
//   p.finish(total_mols);   // prints final line with \n

class Progress {
public:
    Progress(const char* phase, uint64_t total, const char* unit = "items")
        : phase_(phase), total_(total), unit_(unit) {
        t_start_ = t_last_ = std::chrono::steady_clock::now();
        std::fprintf(stderr, "\n[PHASE] %s  (total: %s %s)\n",
                     phase, fmt_human(total).c_str(), unit);
        std::fflush(stderr);
    }

    // Call every iteration; internally throttles to ~10 Hz.
    void update(uint64_t done) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - t_last_).count() < 0.1) return;
        t_last_ = now;
        render(done, now, false);
    }

    // Prints final line with \n.
    void finish(uint64_t done) {
        render(done, std::chrono::steady_clock::now(), true);
    }

private:
    const char* phase_;
    uint64_t    total_;
    const char* unit_;
    std::chrono::steady_clock::time_point t_start_, t_last_;

    void render(uint64_t done,
                const std::chrono::steady_clock::time_point& now,
                bool final) const {
        double elapsed = std::chrono::duration<double>(now - t_start_).count();
        double rate    = (elapsed > 0.05) ? done / elapsed : 0.0;
        double pct     = (total_ > 0) ? 100.0 * done / total_ : 0.0;
        double eta_s   = (rate > 0 && done < total_) ? (total_ - done) / rate : -1.0;

        // 20-char ASCII bar
        char bar[21] = {};
        int filled = (int)(pct / 5.0);
        for (int i = 0; i < 20; i++)
            bar[i] = (i < filled) ? '=' : (i == filled ? '>' : '.');

        char line[320];
        std::snprintf(line, sizeof(line),
            "  [%s] %-34s  %5.1f%%  %8s / %-8s %-5s  |  %8s/s  |  ETA %-12s  |  %s     ",
            bar, phase_,
            pct,
            fmt_human(done).c_str(),
            fmt_human(total_).c_str(),
            unit_,
            fmt_human((uint64_t)(rate + 0.5)).c_str(),
            fmt_eta(eta_s).c_str(),
            fmt_elapsed(elapsed).c_str());

        std::fprintf(stderr, final ? "\r%s\n" : "\r%s", line);
        std::fflush(stderr);
    }
};

// ── ProgressBytes: like Progress but formatted for byte counts ────────────────
class ProgressBytes {
public:
    ProgressBytes(const char* phase, uint64_t total_bytes)
        : phase_(phase), total_(total_bytes) {
        t_start_ = t_last_ = std::chrono::steady_clock::now();
        std::fprintf(stderr, "\n[PHASE] %s  (total: %s)\n",
                     phase, fmt_bytes(total_bytes).c_str());
        std::fflush(stderr);
    }

    void update(uint64_t bytes_done, uint64_t extra_count = 0,
                const char* extra_unit = nullptr) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - t_last_).count() < 0.1) return;
        t_last_ = now;
        render(bytes_done, extra_count, extra_unit, now, false);
    }

    void finish(uint64_t bytes_done, uint64_t extra_count = 0,
                const char* extra_unit = nullptr) {
        render(bytes_done, extra_count, extra_unit,
               std::chrono::steady_clock::now(), true);
    }

private:
    const char* phase_;
    uint64_t    total_;
    std::chrono::steady_clock::time_point t_start_, t_last_;

    void render(uint64_t done, uint64_t extra, const char* extra_unit,
                const std::chrono::steady_clock::time_point& now,
                bool final) const {
        double elapsed = std::chrono::duration<double>(now - t_start_).count();
        double rate    = (elapsed > 0.05) ? done / elapsed : 0.0;
        double pct     = (total_ > 0) ? 100.0 * done / total_ : 0.0;
        double eta_s   = (rate > 0 && done < total_) ? (total_ - done) / rate : -1.0;

        char bar[21] = {};
        int filled = (int)(pct / 5.0);
        for (int i = 0; i < 20; i++)
            bar[i] = (i < filled) ? '=' : (i == filled ? '>' : '.');

        char extra_buf[64] = {};
        if (extra > 0 && extra_unit)
            std::snprintf(extra_buf, sizeof(extra_buf),
                          "  |  %s %s", fmt_human(extra).c_str(), extra_unit);

        char line[320];
        std::snprintf(line, sizeof(line),
            "  [%s] %-30s  %5.1f%%  %8s / %-8s  |  %8s/s  |  ETA %-12s  |  %s%s     ",
            bar, phase_,
            pct,
            fmt_bytes(done).c_str(),
            fmt_bytes(total_).c_str(),
            fmt_bytes((uint64_t)(rate + 0.5)).c_str(),
            fmt_eta(eta_s).c_str(),
            fmt_elapsed(elapsed).c_str(),
            extra_buf);

        std::fprintf(stderr, final ? "\r%s\n" : "\r%s", line);
        std::fflush(stderr);
    }
};

// ── Spinner: for phases with no known total (e.g. FAISS training) ────────────
//
// Usage:
//   Spinner s("Phase 3: training FAISS");
//   while (working) { s.tick(); }       // call frequently; throttled to 5 Hz
//   s.finish("500K vectors, 1024 cells");

class Spinner {
public:
    explicit Spinner(const char* phase)
        : phase_(phase),
          t_start_(std::chrono::steady_clock::now()),
          t_last_ (std::chrono::steady_clock::now()) {
        std::fprintf(stderr, "\n[PHASE] %s\n", phase);
        std::fflush(stderr);
    }

    void tick(const char* extra = nullptr) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - t_last_).count() < 0.2) return;
        t_last_ = now;
        static const char f[] = {'|','/','-','\\'};
        double elapsed = std::chrono::duration<double>(now - t_start_).count();
        char line[256];
        if (extra)
            std::snprintf(line, sizeof(line),
                "  [%c]  %-42s  %s  |  %s     ",
                f[frame_++ % 4], phase_, extra, fmt_elapsed(elapsed).c_str());
        else
            std::snprintf(line, sizeof(line),
                "  [%c]  %-42s  %s     ",
                f[frame_++ % 4], phase_, fmt_elapsed(elapsed).c_str());
        std::fprintf(stderr, "\r%s", line);
        std::fflush(stderr);
    }

    void finish(const char* summary = nullptr) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start_).count();
        char line[256];
        if (summary)
            std::snprintf(line, sizeof(line),
                "  [OK]  %-42s  %s  |  %s     ",
                phase_, summary, fmt_elapsed(elapsed).c_str());
        else
            std::snprintf(line, sizeof(line),
                "  [OK]  %-42s  %s     ",
                phase_, fmt_elapsed(elapsed).c_str());
        std::fprintf(stderr, "\r%s\n", line);
        std::fflush(stderr);
    }

private:
    const char* phase_;
    std::chrono::steady_clock::time_point t_start_, t_last_;
    int frame_ = 0;
};

} // namespace bscs
