// Offline reader for CANN 9.2.1 profiler's compact JSON layout.
// Any complete "X" record using an unknown layout fails closed; Python can validate other layouts.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <openssl/sha.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class FileView {
public:
    explicit FileView(const char *name) {
        int fd = open(name, O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open input");
        struct stat st{};
        if (fstat(fd, &st) == -1 || st.st_size <= 0) {
            close(fd);
            throw std::runtime_error("Empty or unreadable input");
        }
        length = static_cast<size_t>(st.st_size);
        data = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    ~FileView() { munmap(data, length); }
    FileView(const FileView &) = delete;
    FileView &operator=(const FileView &) = delete;
    std::string_view view() const { return {static_cast<const char *>(data), length}; }
private:
    void *data;
    size_t length;
};

static double number(std::string_view s) {
    // The actual JSON delimiter is the first byte after s, so strtod stops inside the mapped file.
    if (s.empty() || s.size() > 40 || s.find_first_not_of("0123456789.-+eE") != s.npos)
        throw std::runtime_error("Invalid timestamp");
    char *end = nullptr;
    errno = 0;
    double x = std::strtod(s.data(), &end);
    if (end != s.data() + s.size() || errno == ERANGE || x < 0)
        throw std::runtime_error("Cannot parse timestamp");
    return x;
}

static void reduce(const char *path) {
    FileView file(path);
    auto b = file.view();
    if (b.back() != '}' && b.back() != '\n') throw std::runtime_error("Missing trace footer");
    constexpr std::string_view ph = "\"ph\":\"X\"";
    constexpr std::string_view dur = "\"dur\":";
    constexpr std::string_view name = ",\"name\":\"";
    constexpr std::string_view ts = ",\"ts\":";
    size_t position = 0, count = 0, end_count = 0, mmads = 0;
    double first = std::numeric_limits<double>::infinity(), last = 0, physical_end = 0;
    while ((position = b.find(ph, position)) != b.npos) {
        auto lead = b.substr(position > 200 ? position - 200 : 0, std::min(position, size_t(200)));
        auto n = lead.rfind(name);
        auto d = lead.rfind(dur, n);
        auto tail = b.substr(position, std::min(b.size() - position, size_t(180)));
        auto t = tail.find(ts, ph.size());
        auto end = tail.find('}', t);
        if (n == b.npos || d == b.npos || d+dur.size()>n || lead.substr(lead.size()-2)!="\"," ||
            t == b.npos || end == b.npos || tail.substr(ph.size(),7)!=",\"pid\":" ||
            n+name.size()>lead.size()-2)
            throw std::runtime_error("Unknown complete-event layout");
        auto op = lead.substr(n+name.size(),lead.size()-2-n-name.size());
        auto start = number(tail.substr(t+ts.size(),end-t-ts.size()));
        auto finish = start + number(lead.substr(d+dur.size(),n-d-dur.size()));
        first=std::min(first,start); last=std::max(last,finish); ++count;
        if (op=="END") { ++end_count; physical_end=start; }
        mmads+=(op=="MMAD"); position+=end+1;
    }
    if (!count || end_count!=1 || physical_end<first || physical_end>last)
        throw std::runtime_error("Missing physical END/retirement");
    unsigned char hash[SHA256_DIGEST_LENGTH]{};
    if (!SHA256(reinterpret_cast<const unsigned char*>(b.data()),b.size(),hash))
        throw std::runtime_error("Cannot fingerprint raw trace");
    std::cout<<std::setprecision(17)<<first<<'\t'<<last<<'\t'<<physical_end<<'\t'<<count<<'\t'<<mmads<<'\t';
    for (auto byte : hash) std::cout<<std::hex<<std::setfill('0')<<std::setw(2)<<static_cast<unsigned>(byte);
    std::cout << std::dec << '\n';
}

int main(int argc,char **argv) {
    if (argc<2) return 2;
    try {
        for (int i=1;i<argc;++i) reduce(argv[i]);
    } catch (const std::exception &e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
    return 0;
}
