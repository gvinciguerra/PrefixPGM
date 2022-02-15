// This file is part of PGM-index <https://github.com/gvinciguerra/PGM-index>.
// Copyright (c) 2021 Giorgio Vinciguerra.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "piecewise_linear_model.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>


#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

/**
 * A struct that stores the result of a query to a @ref PGMIndex, that is, a range [@ref lo, @ref hi)
 * centered around an approximate position @ref pos of the sought key.
 */
struct ApproxPos {
    size_t pos; ///< The approximate position of the key.
    size_t lo;  ///< The lower bound of the range.
    size_t hi;  ///< The upper bound of the range.
};

namespace pgm {

namespace internal {

template<typename K>
static size_t to_c_string(const K &k, std::unique_ptr<char[]> &out) {
    auto &repr = k.crepresentation();
    auto limb_pos = std::find_if(repr.begin(), repr.end(), [](auto &x) { return x != 0; }) - repr.begin();
    constexpr auto size_limb = sizeof(typename K::limb_type);
    auto string_length = (repr.size() - limb_pos) * size_limb;
    if (limb_pos != repr.size())
        string_length -= __builtin_ctzll(repr[limb_pos]) / 8;
    out = std::make_unique<char[]>(string_length + 1);
    for (auto i = 0; i < string_length; ++i) // TODO: optimise
        out[i] = *(repr.end() - i / size_limb - 1) >> (8 * (size_limb - i % size_limb - 1));

    out[string_length] = '\0';
    return string_length;
}

template<typename ToType, typename Iterator>
class cast_iterator {
    Iterator it;

public:
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using reference = ToType;
    using pointer = ToType *;
    using iterator_category = std::random_access_iterator_tag;

    cast_iterator(Iterator it) : it(it) {}
    cast_iterator(const cast_iterator &other) : it(other.it) {}

    cast_iterator &operator=(const cast_iterator &other) {
        it = other.it;
        return *this;
    }

    reference operator*() const { return ToType(*it); }
    reference operator[](difference_type i) const { return (ToType) *(*this + i); }

    cast_iterator &operator++() {
        ++it;
        return *this;
    }

    cast_iterator &operator--() {
        --it;
        return *this;
    }

    cast_iterator operator++(int) {
        cast_iterator it(*this);
        ++(*this);
        return it;
    }

    cast_iterator operator--(int) {
        cast_iterator it(*this);
        --(*this);
        return it;
    }

    cast_iterator operator+(difference_type n) const { return {it + n}; }
    cast_iterator operator-(difference_type n) const { return {it - n}; }

    cast_iterator &operator+=(difference_type n) {
        it += n;
        return *this;
    }

    cast_iterator &operator-=(difference_type n) {
        it -= n;
        return *this;
    }

    difference_type operator-(const cast_iterator &i) const { return it - i.it; }

    bool operator<(const cast_iterator &i) const { return *this - i < 0; }
    bool operator>(const cast_iterator &i) const { return i < *this; }
    bool operator<=(const cast_iterator &i) const { return !(*this > i); }
    bool operator>=(const cast_iterator &i) const { return !(*this < i); }
    bool operator!=(const cast_iterator &i) const { return !(*this == i); }
    bool operator==(const cast_iterator &i) const { return *this - i == 0; }
};

}

template<size_t PrefixBytes = 16, size_t Epsilon = 64, size_t EpsilonRecursive = 4, bool KeepFullPrefix = true>
class PrefixPGMIndex {
protected:
    static_assert(PrefixBytes >= 4);
    static_assert(Epsilon > 0);
    struct SegmentA;
    struct SegmentB;

    using Segment = std::conditional_t<KeepFullPrefix, SegmentA, SegmentB>;

    using K = adapter_uintwide_t<8 * PrefixBytes, std::uint64_t, void, false>;
    using L = internal::LargeUnsigned<K>;

    size_t n;                           ///< The number of elements this index was built on.
    K first_key;                        ///< The smallest element.
    std::vector<Segment> segments;      ///< The segments composing the index.
    std::vector<size_t> levels_offsets; ///< The starting position of each level in segments[], in reverse order.
    static constexpr size_t linear_search_threshold = 64 * 64 / sizeof(Segment);

    template<typename RandomIt>
    static void build(RandomIt first, RandomIt last,
                      size_t epsilon, size_t epsilon_recursive,
                      std::vector<Segment> &segments,
                      std::vector<size_t> &levels_offsets) {
        auto n = (size_t) std::distance(first, last);
        if (n == 0)
            return;

        levels_offsets.push_back(0);
        segments.reserve(n / (epsilon * epsilon));

        auto ignore_last = *std::prev(last) == std::numeric_limits<K>::max(); // max() is the sentinel value
        auto last_n = n - ignore_last;
        last -= ignore_last;

        auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
            auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun);
            if (segments.back().slope_numerator == 0 && last_n > 1) {
                // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
                segments.emplace_back(*std::prev(last) + K(1), last_n);
                ++n_segments;
            }
            segments.emplace_back(last_n); // Add the sentinel segment
            return n_segments;
        };

        // Build first level
        auto in_fun = [&](auto i) {
            auto x = first[i];
            // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
            // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
            auto flag = i > 0 && i + 1u < n && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
            return std::pair<K, size_t>(x + flag, i);
        };
        auto out_fun = [&](auto cs) { segments.emplace_back(cs); };
        last_n = build_level(epsilon, in_fun, out_fun);
        levels_offsets.push_back(levels_offsets.back() + last_n + 1);

        // Build upper levels
        while (epsilon_recursive && last_n > 1) {
            auto offset = levels_offsets[levels_offsets.size() - 2];
            auto in_fun_rec = [&](auto i) { return std::pair<K, size_t>(segments[offset + i].get_key(), i); };
            last_n = build_level(epsilon_recursive, in_fun_rec, out_fun);
            levels_offsets.push_back(levels_offsets.back() + last_n + 1);
        }
    }

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to approximate_position for
     * @return an iterator to the segment responsible for the given key
     */
    auto segment_for_key(const K &key) const {
        if constexpr (EpsilonRecursive == 0) {
            return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
        }

        auto it = segments.begin() + *(levels_offsets.end() - 2);
        for (auto l = int(height()) - 2; l >= 0; --l) {
            auto level_begin = segments.begin() + levels_offsets[l];
            auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
            auto lo = level_begin + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            if constexpr (EpsilonRecursive <= linear_search_threshold) {
                for (; *std::next(lo) <= key; ++lo)
                    continue;
                it = lo;
            } else {
                auto level_size = levels_offsets[l + 1] - levels_offsets[l] - 1;
                auto hi = level_begin + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);
                it = std::prev(std::upper_bound(lo, hi, key));
            }
        }

        return it;
    }

    template<typename Iterator>
    internal::cast_iterator<K, Iterator> make_cast_iterator(Iterator it) { return {it}; }

public:

    static constexpr size_t epsilon_value = Epsilon;

    /**
     * Constructs an empty index.
     */
    PrefixPGMIndex() = default;

    /**
     * Constructs the index on the given sorted vector.
     * @param data the vector of keys to be indexed, must be sorted
     */
    explicit PrefixPGMIndex(const std::vector<K> &data) : PrefixPGMIndex(data.begin(), data.end()) {}

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    PrefixPGMIndex(RandomIt first, RandomIt last)
        : n(std::distance(first, last)),
          first_key(n ? K(*first) : K(0)),
          segments(),
          levels_offsets() {
        build(make_cast_iterator(first), make_cast_iterator(last), Epsilon, EpsilonRecursive, segments, levels_offsets);
    }

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to approximate_position for
     * @return a struct with the approximate position and bounds of the range
     */
    ApproxPos approximate_position(std::string_view string_key) const {
        K k(string_key);
        auto it = segment_for_key(k);
        auto pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
        auto lo = PGM_SUB_EPS(pos, Epsilon);
        auto hi = PGM_ADD_EPS(pos, Epsilon, n);
        return {pos, lo, hi};
    }

    template<typename RandomIt>
    RandomIt lower_bound(RandomIt first, RandomIt last, std::string_view string_key) {
        auto range = approximate_position(string_key);
        auto lo = first + range.lo;
        auto hi = first + range.hi;
        auto it = std::lower_bound(lo, hi, string_key);
        for (; it < last && *it < string_key ; ++it)
            continue;
        return it;
    }

    template<typename RandomIt>
    RandomIt upper_bound(RandomIt first, RandomIt last, std::string_view string_key) {
        auto range = approximate_position(string_key);
        auto lo = first + range.lo;
        auto hi = first + range.hi;
        auto it = std::upper_bound(lo, hi, string_key);
        for (; it < last && *it <= string_key ; ++it)
            continue;
        return it;
    }

    /**
     * Returns the number of segments in the last level of the index.
     * @return the number of segments
     */
    size_t segments_count() const { return segments.empty() ? 0 : levels_offsets[1] - 1; }

    /**
     * Returns the number of levels of the index.
     * @return the number of levels of the index
     */
    size_t height() const { return levels_offsets.size() - 1; }

    /**
     * Returns the size of the index in bytes.
     * @return the size of the index in bytes
     */
    size_t size_in_bytes() const {
        size_t sum = 0;
        for (auto &s: segments)
            sum += s.size_in_bytes();
        return sum + levels_offsets.size() * sizeof(size_t);
    }
};

#pragma pack(push, 1)

template<size_t PrefixBytes, size_t Epsilon, size_t EpsilonRecursive, bool KeepFullPrefix>
struct PrefixPGMIndex<PrefixBytes, Epsilon, EpsilonRecursive, KeepFullPrefix>::SegmentA {
    K key;
    uint32_t slope_numerator;
    K slope_denominator;
    int32_t intercept;

    SegmentA() = default;

    template<typename I>
    SegmentA(K key, I intercept) : key(key), slope_numerator(), slope_denominator(), intercept(int32_t(intercept)) {};

    explicit SegmentA(size_t n)
        : key(std::numeric_limits<K>::max()), slope_numerator(), slope_denominator(), intercept(n) {};

    explicit SegmentA(const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs) {
        auto[cs_numerator, cs_denominator, cs_intercept] = cs.get_segment_parameters();
        if (cs_intercept > (decltype(cs_intercept)) std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        key = cs.get_first_x();
        slope_numerator = (decltype(slope_numerator)) cs_numerator;
        slope_denominator = (decltype(slope_denominator)) cs_denominator;
        intercept = (decltype(intercept)) cs_intercept;
    }

    friend inline bool operator<(const SegmentA &s, const K &k) { return s.key < k; }
    friend inline bool operator<=(const SegmentA &s, const K &k) { return s.key <= k; }
    friend inline bool operator<(const K &k, const SegmentA &s) { return k < s.key; }
    friend inline bool operator<(const SegmentA &s, const SegmentA &t) { return s.key < t.key; }

    K get_key() const { return key; }

    inline size_t operator()(const K &k) const {
        auto result = slope_numerator * L(k - key) / L(slope_denominator);
        if (__builtin_expect(result > std::numeric_limits<int64_t>::max(), 0))
            return -1;
        auto pos = int64_t(result) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }

    size_t size_in_bytes() const { return sizeof(*this); }
};

template<size_t PrefixBytes, size_t Epsilon, size_t EpsilonRecursive, bool KeepFullPrefix>
struct PrefixPGMIndex<PrefixBytes, Epsilon, EpsilonRecursive, KeepFullPrefix>::SegmentB {
    std::unique_ptr<char[]> key;
    uint32_t slope_numerator;
    std::unique_ptr<char[]> slope_denominator;
    int16_t slope_denominator_length;
    int32_t intercept;

    SegmentB() = default;

    template<typename I>
    SegmentB(K k, I intercept)
        : key(),
          slope_numerator(),
          slope_denominator(),
          slope_denominator_length(),
          intercept(int32_t(intercept)) {
        internal::to_c_string(k, key);
    }

    explicit SegmentB(size_t n)
        : key(),
          slope_numerator(),
          slope_denominator(),
          slope_denominator_length(),
          intercept(n) {
        internal::to_c_string(std::numeric_limits<K>::max(), key);
    }

    explicit SegmentB(const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs) {
        auto[cs_numerator, cs_denominator, cs_intercept] = cs.get_segment_parameters();
        if (cs_intercept > (decltype(cs_intercept)) std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        internal::to_c_string(cs.get_first_x(), key);
        assert(K(key.get()) == cs.get_first_x());
        slope_numerator = (decltype(slope_numerator)) cs_numerator;
        slope_denominator_length = internal::to_c_string(cs_denominator, slope_denominator);
        assert(K(std::string_view(slope_denominator.get(), slope_denominator_length)) == cs_denominator);
        intercept = (decltype(intercept)) cs_intercept;
    }

    friend inline bool operator<(const SegmentB &s, const K &k) { return K(s.key.get()) < k; }
    friend inline bool operator<=(const SegmentB &s, const K &k) { return K(s.key.get()) <= k; }
    friend inline bool operator<(const K &k, const SegmentB &s) { return k < K(s.key.get()); }
    friend inline bool operator<(const SegmentB &s, const SegmentB &t) { return strcmp(s.key.get(), t.key.get()); }

    K get_key() const { return K(key.get()); }

    inline size_t operator()(const K &k) const {
        auto l_key = get_key();
        auto l_den = L(K(std::string_view(slope_denominator.get(), slope_denominator_length)));
        auto pos = int64_t(slope_numerator * L(k - l_key) / l_den) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }

    size_t size_in_bytes() const { return std::strlen(key.get()) + 1 + slope_denominator_length + sizeof(*this); }
};

#pragma pack(pop)

}