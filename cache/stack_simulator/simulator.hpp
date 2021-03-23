#ifndef __SIMULATOR_HPP__
#define __SIMULATOR_HPP__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

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
    std::map<uint64_t, float> hitrates(const std::vector<T>& requests);
    std::map<uint64_t, float> hitrates(const std::vector<T>& requests,
        const std::vector<int>& cache_sizes);
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

void export_csv(
    const std::vector<std::map<uint64_t, float>>& hitrates,
    const std::string& basename,
    const int feature_offset);

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
std::map<uint64_t, float>
Simulator<Policy::LRU, T>::hitrates(const std::vector<T>& requests, const std::vector<int>& cache_sizes)
{
    
}

template<typename T>
std::map<uint64_t, float>
Simulator<Policy::LFU, T>::hitrates(
    const std::vector<T>& requests,
    const std::vector<uint64_t>& cache_sizes)
{
    std::map<uint64_t, float> hrates;
    using heap_t = std::pair<uint64_t, T>;
    const float n_requests = static_cast<float>(requests.size());

    for (const auto& cache_max_size : cache_sizes)
    {
        std::vector<heap_t> in_cache;
        
        in_cache.reserve(cache_max_size);
        uint64_t hits = 0;

        for (const auto& req : requests)
        {
            const auto elem = std::find_if(in_cache.begin(), in_cache.end(),
                    [&](const auto& p) { return p.second == req; });

            if (elem != in_cache.end()) // cache hit
            {
                ++(elem->first);
                std::make_heap(in_cache.begin(), in_cache.end(),
                    std::greater<heap_t>());
                ++hits;
            }
            else // cache miss
            {
                if (in_cache.size() == cache_max_size) // eviction
                {
                    std::pop_heap(in_cache.begin(), in_cache.end());
                    in_cache.pop_back();
                }
                in_cache.push_back({1, req});
                std::push_heap(in_cache.begin(), in_cache.end(),
                    std::greater<heap_t>());
            }
            
        }
        hrates[cache_max_size] = static_cast<float>(hits) / n_requests;
    }

    return hrates;
}

void export_csv(
    const std::vector<std::map<uint64_t, float>>& hitrates,
    const std::string& basename,
    const int feature_offset)
{
	const int N = hitrates.size();
	for (int i = 0; i<N; i++)
	{
		std::string name =
            basename + std::to_string(feature_offset + i) + ".csv";
		std::ofstream file(name, std::ofstream::binary);
		file.precision(5);
		file << "size,hitrate\n";

		for (const auto& [key, val] : hitrates[i])
			file << key << ',' << val << '\n';
		
		file << endl;
		file.flush();
		file.close();
	}
}

}

#endif // #ifndef __SIMULATOR_HPP__