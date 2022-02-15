#include <iostream>
#include <fstream>
#include <string>
#include "pgm_index_prefix.hpp"

std::vector<std::string> read_string_prefixes(const std::string &path, size_t prefix_length, size_t limit = -1) {
    auto previous_value = std::ios::sync_with_stdio(false);
    std::vector<std::string> result;
    std::ifstream in(path.c_str());
    std::string str;
    while (std::getline(in, str) && limit-- > 0) {
        if (str.size() > prefix_length)
            str.resize(prefix_length);
        result.push_back(str);
    }
    std::ios::sync_with_stdio(previous_value);
    return result;
}

template<int First, int Last, typename Lambda>
inline void static_for_pow_two(Lambda const &f) {
    if constexpr (First <= Last) {
        { f(std::integral_constant<int, First>{}); }
        static_for_pow_two<First << 1, Last>(f);
    }
}

int main() {
    constexpr auto prefix_size = 16;
    std::vector<std::string> data = read_string_prefixes("/usr/share/dict/words", prefix_size);
    std::cout << "Read " << data.size() << " lines" << std::endl;
    std::sort(data.begin(), data.end());
    auto new_end = std::unique(data.begin(), data.end());
    if (new_end != data.end()) {
        data.erase(new_end, data.end());
        std::cout << "After removing duplicates " << data.size() << std::endl;
    }

    static_for_pow_two<8, 64>([&](auto epsilon) {
        std::cout << std::string(79, '-') << std::endl;

        // -------------- PGM CONSTRUCTION --------------
        std::cout << "Prefix size " << prefix_size << ", epsilon " << epsilon << std::endl;

        pgm::PrefixPGMIndex<prefix_size, epsilon, 0, true> pgm(data.begin(), data.end());
        std::cout << "PGM #segm: " << pgm.segments_count() << std::endl;
        std::cout << "PGM bytes: " << pgm.size_in_bytes() << std::endl;

        // -------------- TEST --------------
        for (size_t i = 0; i < data.size(); ++i) {
            auto range = pgm.approximate_position(data[i]);
            if (i > range.hi || i < range.lo) {
                std::cout << data[i] << " " << i << " vs returned " << range.lo << " " << range.hi << std::endl;
                exit(1);
            }
        }
        std::cout << "Tests done" << std::endl;
    });
    return 0;
}
