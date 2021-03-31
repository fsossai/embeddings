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

#include "fixed_size_heap.hpp"
#include "rank_tree.hpp"

namespace cache {

class Policy
{
public:
    class LRU;
    class LFU;
    class OPT;
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


template<typename T>
class Simulator<Policy::OPT, T>
{
public:
    Simulator() = default;

    std::map<uint64_t, float> hitrates(const std::vector<T>& requests,
        const std::vector<uint64_t>& cache_sizes);

private:
    std::unordered_map<T, std::vector<uint64_t>> next_ref_times(
        const std::vector<T>& requests
    );
};


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

    for (uint64_t cache_max_size : cache_sizes)
    {
        if (cache_max_size == 0)
        {
            hrates[0] = 0.0f;
            continue;
        }

        uint64_t hits = 0;
        FixedSizeHeap<T, uint64_t, std::less<uint64_t>> cache(cache_max_size);

        for (const T& req : requests)
        {
            if (cache.contains(req)) // cache hit
            {
                cache.set(req, cache.get(req) + 1);
                ++hits;
            }
            else // cache miss
            {
                cache.insert(req, 1);
            }
        }
        
        hrates[cache_max_size] = static_cast<float>(hits) / n_requests;
    }

    return hrates;
}

/*** OPT ***/

template<typename T>
std::map<uint64_t, float>
Simulator<Policy::OPT, T>::hitrates(
    const std::vector<T>& requests,
    const std::vector<uint64_t>& cache_sizes)
{
    std::map<uint64_t, float> hrates;
    const float n_requests = static_cast<float>(requests.size());

    for (uint64_t cache_max_size : cache_sizes)
    {
        if (cache_max_size == 0)
        {
            hrates[0] = 0.0f;
            continue;
        }

        uint64_t hits = 0;
        FixedSizeHeap<T, uint64_t, std::greater<uint64_t>> cache(cache_max_size);
        std::unordered_map<T, int> next_ref_time_index;
        auto next = next_ref_times(requests);

        for (const T& req : requests)
        {
            auto req_next_access = next[req][++next_ref_time_index[req]];
            if (cache.contains(req)) // cache hit
            {
                cache.set(req, std::move(req_next_access));
                ++hits;
            }
            else // cache miss
            {
                cache.insert(req, req_next_access);
            }
        }
        
        hrates[cache_max_size] = static_cast<float>(hits) / n_requests;
    }

    return hrates;
}


template<typename T>
std::unordered_map<T, std::vector<uint64_t>>
Simulator<Policy::OPT, T>::next_ref_times(const std::vector<T>& requests)
{
    std::unordered_map<T, std::vector<uint64_t>> ref_times;

    for (uint64_t i = 0; i < requests.size(); ++i)
    {
        auto&& req = requests[i];
        if (ref_times.find(req) != ref_times.end())
        {
            ref_times[req].push_back(i);
        }
        else
        {
            ref_times.insert({req, std::vector<uint64_t>{i}});
        }
    }

    return ref_times;
}

} // end namespace cache

#endif // #ifndef __SIMULATOR_HPP__