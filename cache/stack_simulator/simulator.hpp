#ifndef __SIMULATOR_HPP__
#define __SIMULATOR_HPP__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "rank_tree.hpp"

namespace cache {

class Policy
{
public:
    class LRU;
    class LFU;
};

template<typename P, typename T>
class Simulator
{ };

template<typename T>
class Simulator<Policy::LRU, T>
{
public:
    Simulator() = default;

    uint64_t reference(const T& key);

    std::map<uint64_t, float>
        hitrates(const std::vector<T>& requests);

    std::map<std::pair<uint64_t, float>, float>
        hitrates_relative(const std::vector<T>& requests);

    std::map<std::pair<uint64_t, float>, float>
        hitrates_relative(
            const std::vector<T>& requests,
            const uint64_t nunique
        );

    std::map<std::pair<uint64_t, float>, float>
        hitrates_relative(
            const std::vector<T>& requests,
            const std::vector<float>& cache_sizes_percentage
        );

    std::map<std::pair<uint64_t, float>, float>
        hitrates_relative(
            const std::vector<T>& requests,
            const std::vector<float>& cache_sizes_percentage,
            const uint64_t nunique
        );

private:
    RankTree<T> _rtree;
    std::unordered_map<T, RankTreeNode<T>*> _umap;
};

template<typename T>
class Simulator<Policy::LFU, T>
{
public:
    Simulator() = default;
    
    std::map<uint64_t, float> hitrates(const std::vector<T>& requests,
        const std::vector<uint64_t>& cache_sizes);

private:
    //
};

template<typename ForwardIt, typename Getter>
ForwardIt min_positive_int(
    ForwardIt first,
    ForwardIt last,
    const int limit,
    Getter p);

/*** DEFINITIONS ***/

template<typename T>
uint64_t Simulator<Policy::LRU, T>::reference(const T& key)
{
    if (_umap.count(key) == 0)
    {
		_umap[key] = _rtree.Insert(key);
		return UINT64_MAX;
    }
    auto node = _umap[key];
    uint64_t rank = node->Rank();
    _rtree.Remove(node);
    _rtree.InsertNode(node);
    return rank;
}


template<typename T>
std::map<uint64_t, float>
Simulator<Policy::LRU, T>::hitrates(const std::vector<T>& requests)
{
    std::vector<uint64_t> stack_distances;
    std::map<uint64_t, float> hrates;
    stack_distances.reserve(requests.size());

    for (const auto& key : requests)
        stack_distances.push_back(reference(key));

    std::sort(stack_distances.begin(), stack_distances.end());

    const float length = static_cast<float>(stack_distances.size());
    int key = 0, count = 0;
    for (const uint64_t val : stack_distances)
    {
        if (val != key)
        {
            hrates[key + 1] = static_cast<float>(count) / length;
            key = val;
        }
        ++count;
    }

    return hrates;
}


template<typename T>
std::map<std::pair<uint64_t, float>, float>
Simulator<Policy::LRU, T>::hitrates_relative(
    const std::vector<T>& requests)
{
    // obtaining the number of unique values
    const int N = std::unordered_set<T>(
        requests.begin(), requests.end()).size();
    return hitrates_relative(requests, N);
}


template<typename T>
std::map<std::pair<uint64_t, float>, float>
Simulator<Policy::LRU, T>::hitrates_relative(
    const std::vector<T>& requests,
    const uint64_t nunique)
{
    std::vector<uint64_t> stack_distances;
    std::map<std::pair<uint64_t, float>, float> hrates;
    stack_distances.reserve(requests.size());

    // this is the core of the simulation
    for (const auto& key : requests)
        stack_distances.push_back(reference(key));

    std::sort(stack_distances.begin(), stack_distances.end());

    const float length = static_cast<float>(stack_distances.size());
    int key = 0, count = 0;
    for (const auto& val : stack_distances)
    {
        const std::pair<uint64_t, float> keys = {key, (key + 1.0f) / nunique};
        if (val != key)
        {
            hrates[keys] = static_cast<float>(count) / length;
            key = val;
        }
        ++count;
    }

    return hrates;
}

template<typename T>
std::map<std::pair<uint64_t, float>, float>
Simulator<Policy::LRU, T>::hitrates_relative(
    const std::vector<T>& requests,
    const std::vector<float>& cache_sizes_percentage)
{
    // obtaining the number of unique values
    const int N = std::unordered_set<T>(
        requests.begin(), requests.end()).size();
    return hitrates_relative(requests, cache_sizes_percentage, N);
}


template<typename T>
std::map<std::pair<uint64_t, float>, float>
Simulator<Policy::LRU, T>::hitrates_relative(
    const std::vector<T>& requests,
    const std::vector<float>& cache_sizes_percentage,
    const uint64_t nunique)
{
    using hitrate_t = std::map<std::pair<uint64_t, float>, float>;
    const hitrate_t hrates = hitrates_relative(requests, nunique);
    hitrate_t filtered;

    std::vector<uint64_t> cache_sizes(cache_sizes_percentage.size());
    std::transform(
        cache_sizes_percentage.begin(),
        cache_sizes_percentage.end(),
        cache_sizes.begin(),
        [nunique](auto s) { return static_cast<uint64_t>(s * nunique); }
    );

    for (const auto cache_max_size : cache_sizes)
    {
        const auto match = std::adjacent_find(
            hrates.begin(), hrates.end(),
            [cache_max_size](const auto& a, const auto& b)
            {
                return (a.first.first <= cache_max_size) &&
                    (cache_max_size < b.first.first);
            }
        );
        const auto [key, val] = *match;
        filtered[key] = val;
    }
    return filtered;
}

/*** LFU ***/

template<typename T>
std::map<uint64_t, float>
Simulator<Policy::LFU, T>::hitrates(
    const std::vector<T>& requests,
    const std::vector<uint64_t>& cache_sizes)
{
    std::map<uint64_t, float> hrates;
    const float n_requests = static_cast<float>(requests.size());

    for (const auto& cache_max_size : cache_sizes)
    {
        if (cache_max_size == 0)
        {
            hrates[0] = 0.0f;
            continue;
        }

        std::unordered_map<T, int> in_cache;
        uint64_t current_size = 0;
        uint64_t hits = 0;
        for (const auto& req : requests)
        {
            if (in_cache[req] > 0) // cache hit
            {
                ++in_cache[req];
                ++hits;
            }
            else // cache miss
            {
                if (current_size == cache_max_size) // eviction
                {
                    auto min_el = min_positive_int(
                        in_cache.begin(),
                        in_cache.end(),
                        std::numeric_limits<int>::max(),
                        [](const auto& elem) { return elem.second; }
                    );
                    min_el->second = 0;
                }
                else
                {
                    ++current_size;
                }
                in_cache[req] = 1;
            }
        }
        hrates[cache_max_size] = static_cast<float>(hits) / n_requests;
    }

    return hrates;
}

// This function is tailored for containers with small values,
// where generally, the smallest positive is 1 or 2.
template<typename ForwardIt, typename Getter>
ForwardIt min_positive_int(
    ForwardIt first,
    ForwardIt last,
    const int limit,
    Getter g)
{
    int key = 1;
    ForwardIt min_el;
    do
    {
        min_el = std::find_if(first, last,
            [&](const auto& elem)
            { return g(elem) == key; }
        );
        ++key;

        if (key == limit)
            return last;
    }
    while (min_el == last);
    return min_el;
}

} // end namespace cache

#endif // #ifndef __SIMULATOR_HPP__