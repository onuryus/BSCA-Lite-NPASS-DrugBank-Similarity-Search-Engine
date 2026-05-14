#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bscs {

// OffsetWriter: streams uint64_t byte offsets to a binary file during indexing.
class OffsetWriter {
public:
    explicit OffsetWriter(const std::string& path) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) throw std::runtime_error("OffsetWriter: cannot open " + path);
    }
    ~OffsetWriter() { if (f_) std::fclose(f_); }

    void write(uint64_t offset) {
        std::fwrite(&offset, sizeof(uint64_t), 1, f_);
        ++count_;
    }
    uint64_t count() const { return count_; }
    void flush() { std::fflush(f_); }

private:
    std::FILE* f_ = nullptr;
    uint64_t count_ = 0;
};

// OffsetStore: memory-mapped read-only offset file. O(1) lookup by molecule ID.
class OffsetStore {
public:
    OffsetStore() = default;

    explicit OffsetStore(const std::string& path) { open(path); }

    ~OffsetStore() { close(); }

    OffsetStore(const OffsetStore&) = delete;
    OffsetStore& operator=(const OffsetStore&) = delete;

    void open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("OffsetStore: cannot open " + path);

        struct stat st{};
        ::fstat(fd_, &st);
        size_bytes_ = static_cast<size_t>(st.st_size);
        count_ = size_bytes_ / sizeof(uint64_t);

        if (size_bytes_ > 0) {
            data_ = static_cast<const uint64_t*>(
                ::mmap(nullptr, size_bytes_, PROT_READ, MAP_SHARED, fd_, 0));
            if (data_ == MAP_FAILED) {
                data_ = nullptr;
                ::close(fd_); fd_ = -1;
                throw std::runtime_error("OffsetStore: mmap failed for " + path);
            }
        }
    }

    void close() {
        if (data_ && size_bytes_) ::munmap(const_cast<uint64_t*>(data_), size_bytes_);
        if (fd_ >= 0) ::close(fd_);
        data_ = nullptr; size_bytes_ = 0; count_ = 0; fd_ = -1;
    }

    uint64_t operator[](uint64_t mol_id) const {
        if (mol_id >= count_)
            throw std::out_of_range("OffsetStore: mol_id out of range");
        return data_[mol_id];
    }

    uint64_t count() const { return count_; }
    bool valid() const { return data_ != nullptr || count_ == 0; }

private:
    const uint64_t* data_ = nullptr;
    size_t size_bytes_ = 0;
    uint64_t count_    = 0;
    int fd_            = -1;
};

} // namespace bscs
