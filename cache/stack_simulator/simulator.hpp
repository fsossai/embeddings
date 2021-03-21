#ifndef __SIMULATOR_HPP__
#define __SIMULATOR_HPP__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include "rank_tree.hpp"

namespace cache {

class Policy
{
public:
    class LRU;
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
private:
    RankTree<T> _rtree;
    std::unordered_map<T, RankTreeNode<T>*> _umap;
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