#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstdio>
#include <chrono>
#include <map>
#include "dataset.hpp"
#include "chronometer.hpp"
#include "cpolicies.hpp"

using std::endl;
using selection_t = std::vector<int>;

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param);

int main(int argc, char** argv)
{
	Chronometer chronometer{};
	constexpr int N_SPARSE_FEATURES = 26;
	constexpr int SPARSE_FEAT_OFFSET = 14;
	selection_t selected_columns(N_SPARSE_FEATURES);
	std::iota(selected_columns.begin(), selected_columns.end(), SPARSE_FEAT_OFFSET);
	
	parser_parameters_t param = {
		.selected_columns = selected_columns,
		.max_samples = 0,
		.separator = '\t',
		.filename = ""
	};
	if (!set_cmdline_args(argc, argv, param))
		return -1;

	Dataset<N_SPARSE_FEATURES> dataset(param);
	chronometer.start();
	std::cout << "Reading dataset ... " << std::flush;
	dataset.import();
	auto samples = dataset.get_samples();
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Calculating LRU stack distances ... " << std::flush;
	auto distances = stack_distances(samples);
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Sorting data ... " << std::flush;
	for (auto& dist : distances)
		std::sort(dist.begin(), dist.end());
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Calculating hit-rate curves ... " << std::flush;
	auto hitrates = hitrate_LRU(distances);
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Exporting to CSV files ... " << std::flush;
	export_csv(hitrates, "hitrates_", SPARSE_FEAT_OFFSET);
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Total time: " << chronometer.elapsed() << endl;

	return 0;
}

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param)
{
	if (argc == 1)
	{
		std::cout << "Error: specify input file" << endl;
		return false;
	}

	if (argc >= 2)
	{
		param.filename = argv[1];
		param.max_samples = std::numeric_limits<int>::max(); 
	}
	if (argc >= 3)
	{
		param.max_samples = std::stoi(argv[2]);
	}

	return true;
}

