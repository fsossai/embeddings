#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstdio>
#include <chrono>
#include <map>
#include <stack-simulator.hpp>
#include "dataset_utils.hpp"

using std::endl;
using selection_t = std::vector<int>;

std::vector<std::vector<int64_t>> stack_distances(const dataset_t& dataset);

int main(int argc, char** argv)
{
	auto timer_start = std::chrono::steady_clock::now();
	const int N_SPARSE_FEATURES = 26;
	const int SPARSE_FEAT_OFFSET = 14;
	selection_t selected_columns(N_SPARSE_FEATURES);
	std::iota(selected_columns.begin(), selected_columns.end(), SPARSE_FEAT_OFFSET);
	parser_parameters_t param = {
		.selected_columns = selected_columns,
		.max_samples = (int)1e6,
		.separator = '\t',
		.filename = "..\\..\\data\\day_1M.csv"
	};

	auto dataset = import_dataset(param);	
	//print_dataset(dataset);

	auto distances = stack_distances(dataset);

	// sorting
	for (auto& dist : distances)
	{
		std::sort(dist.begin(), dist.end());
	}
	
	if (false) // print
	{
		for (int i = 0; i<N_SPARSE_FEATURES; i++)
		{
			std::printf("F%i: ", i);
			for (auto& val : distances[i])
				std::cout << val << ',';
			std::cout << endl;
		}
		std::cout << endl;
	}

	// creating hit-rate curve
	using hrc_t = std::map<int, float>;
	hrc_t *hrcs = new hrc_t[N_SPARSE_FEATURES];
	for (int i = 0; i<N_SPARSE_FEATURES; i++)
	{
		float length = static_cast<float>(distances[i].size());
		int key = 0, count = 0;
		for (int val : distances[i])
		{
			if (val != key)
			{
				hrcs[i][key + 1] = static_cast<float>(count) / length;
				key = val;
			}
			count++;
		}
	}

	// print to csv files, one for each sparse feature
	std::string basename = "hrt_f";
	for (int i = 0; i<N_SPARSE_FEATURES; i++)
	{
		std::string name = basename + std::to_string(SPARSE_FEAT_OFFSET + i) + ".csv";
		std::ofstream file(name, std::ofstream::binary);
		file.precision(3);
		file << "size,hitrate\n";
		std::map<int, float>::iterator it;
		for (it = hrcs[i].begin(); it != hrcs[i].end(); it++)
		{
			file << it->first << ',' << it->second << '\n';
		}
		file << endl;
		file.flush();
		file.close();
	}
	
	auto timer_stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> elasped_sec = timer_stop - timer_start;
	std::cout << "Elapsed time: " << elasped_sec.count() << endl;
	delete[] hrcs;

	return 0;
}

std::vector<std::vector<int64_t>> stack_distances(const dataset_t& dataset)
{
	const int N = dataset[0].size();
	std::vector<std::vector<int64_t>> distances(N);
	StackSimulator *simulators = new StackSimulator[N];

	// initialization
	for (int i = 0; i<N; i++)
	{
		simulators[i] = StackSimulator();
		distances[i] = std::vector<int64_t>();
	}

	for (const auto& sample : dataset)
	{
		for (int i = 0; i<N; i++)
			distances[i].push_back(simulators[i].Reference(sample[i]));
	}

	delete[] simulators;
	return distances;
}

