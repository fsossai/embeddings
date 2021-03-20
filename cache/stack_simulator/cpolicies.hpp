#include <map>
#include <vector>
#include <array>
#include <fstream>
#include "stack-simulator.hpp"

using hrc_t = std::map<int, float>;

template<std::size_t N>
std::array<hrc_t, N> hitrate_LRU(
    const std::array<std::vector<int64_t>, N>& distances);

template<std::size_t N>
std::array<std::vector<int64_t>, N> stack_distances(
    const std::vector<std::array<std::string, N>>& dataset);

template<std::size_t N>
void export_csv(
    const std::array<hrc_t, N>& hrcs,
    const std::string& basename,
    const int feature_offset);


/*** DEFINITIONS ***/

template<std::size_t N>
std::array<hrc_t, N> hitrate_LRU(
    const std::array<std::vector<int64_t>, N>& distances)
{
	std::array<hrc_t, N> hitrates;
	for (int i = 0; i<N; i++)
	{
		const float length = static_cast<float>(distances[i].size());
		int key = 0, count = 0;
		for (int val : distances[i])
		{
			if (val != key)
			{
				hitrates[i][key + 1] = static_cast<float>(count) / length;
				key = val;
			}
			count++;
		}
	}

    return hitrates;
}

template<std::size_t N>
std::array<std::vector<int64_t>, N> stack_distances(
    const std::vector<std::array<std::string, N>>& samples)
{
	std::array<std::vector<int64_t>, N> distances;
    std::array<StackSimulator, N> simulators;

	// initialization
	for (int i = 0; i<N; i++)
	{
		simulators[i] = StackSimulator();
		distances[i] = std::vector<int64_t>();
	}

	for (const auto& sample : samples)
	{
		for (int i = 0; i<N; i++)
			distances[i].push_back(simulators[i].Reference(sample[i]));
	}

	return distances;
}

template<std::size_t N>
void export_csv(
    const std::array<hrc_t, N>& hitrates,
    const std::string& basename,
    const int feature_offset)
{
	for (int i = 0; i<N; i++)
	{
		std::string name =
            basename + std::to_string(feature_offset + i) + ".csv";
		std::ofstream file(name, std::ofstream::binary);
		file.precision(4);
		file << "size,hitrate\n";
		std::map<int, float>::iterator it;
		for (const auto& [key, val] : hitrates[i])
		{
			file << key << ',' << val << '\n';
		}
		file << endl;
		file.flush();
		file.close();
	}
}
