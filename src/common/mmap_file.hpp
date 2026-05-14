#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bscs {

// Read-only memory-mapped file. Zero-copy access to large files.
class MmapFile {
public:
    MmapFile() = default;

    explicit MmapFile(const std::string& path) { open(path); }

    ~MmapFile() { close(); }

    // Non-copyable, movable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& o) noexcept
        : data_(o.data_), size_(o.size_), fd_(o.fd_) {
        o.data_ = nullptr; o.size_ = 0; o.fd_ = -1;
    }

    void open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("MmapFile: cannot open " + path);

        struct stat st{};
        if (::fstat(fd_, &st) < 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("MmapFile: fstat failed for " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) return;

        data_ = static_cast<const char*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("MmapFile: mmap failed for " + path);
        }
        ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
    }

    void close() {
        if (data_ && size_) { ::munmap(const_cast<char*>(data_), size_); }
        if (fd_ >= 0)       { ::close(fd_); }
        data_ = nullptr; size_ = 0; fd_ = -1;
    }

    // Hint random access pattern (for retrieval phase)
    void hint_random() const {
        if (data_ && size_)
            ::madvise(const_cast<char*>(data_), size_, MADV_RANDOM);
    }

    const char* data() const { return data_; }
    size_t      size() const { return size_; }
    bool        valid() const { return data_ != nullptr || size_ == 0; }

    // Read a line starting at byte_offset. Returns string_view-like span.
    std::string read_line(uint64_t byte_offset) const {
        if (byte_offset >= size_) return {};
        const char* start = data_ + byte_offset;
        const char* end   = static_cast<const char*>(
            ::memchr(start, '\n', size_ - byte_offset));
        if (!end) end = data_ + size_;
        // Strip trailing \r
        if (end > start && *(end-1) == '\r') --end;
        return std::string(start, end);
    }

private:
    const char* data_ = nullptr;
    size_t      size_ = 0;
    int         fd_   = -1;
};

} // namespace bscs
