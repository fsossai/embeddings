#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstdio>
#include <map>
#include <stack-simulator.hpp>
#include "dataset_utils.hpp"

using std::endl;
using selection_t = std::vector<int>;

std::vector<std::vector<int64_t>> stack_distances(const dataset_t& dataset);

int main(int argc, char** argv)
{
	const int N_SPARSE_FEATURES = 26;
	selection_t selected_columns(N_SPARSE_FEATURES);
	std::iota(selected_columns.begin(), selected_columns.end(), 14);
	parser_parameters_t param = {
		.selected_columns = selected_columns,
		.max_samples = (int)1e3*0+10,
		.separator = '\t',
		.filename = "..\\..\\data\\day_100k.csv"
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

	// print to csv file
	;

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

