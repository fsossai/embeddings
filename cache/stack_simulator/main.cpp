#include <iostream>
#include <vector>

#include "chronometer.hpp"
#include "dataset.hpp"
#include "simulator.hpp"
#include "utilities.hpp"

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param);

int main(int argc, char** argv)
{
	using namespace cache;

	Chronometer chronometer;
	const int N = 8;
	const int SPARSE_OFFSET = 14;


	const std::vector<float> cache_sizes_percentage{
		0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25 };

	std::vector<int> selected_columns{33, 14, 35, 23, 34, 24, 36, 25};

	parser_parameters_t param = {
		.selected_columns = selected_columns,
		.max_samples = 1,
		.separator = '\t',
		.filename = ""
	};

	if (!set_cmdline_args(argc, argv, param))
		return -1;

	chronometer.start();
	std::cout << "Reading dataset ... " << std::flush;
	ColMajorDataset<N> dataset(param);
	dataset.import();
	auto features = dataset.get_features();
	std::cout << chronometer.lap() << "s" << std::endl;


	/// LRU Policy
	std::cout << "LRU: calculating hitrate curves ... " << std::flush;

	std::vector<std::map<std::pair<uint64_t, float>, float>> hitrates_LRU;
	std::vector<std::vector<uint64_t>> cache_sizes;
	cache_sizes.reserve(N);
	hitrates_LRU.reserve(N);
	for (const auto& f : features)
	{
		Simulator<Policy::LRU, std::string> simulator;
		const auto hrates = simulator.hitrates_relative(f, cache_sizes_percentage);
		hitrates_LRU.push_back(hrates);
		cache_sizes.push_back(get_sizes(hrates));
	}
	std::cout << chronometer.lap() << "s" << std::endl;


	/// LFU Policy
	std::cout << "LFU: calculating hitrate curves ... " << std::flush;

	std::vector<std::map<uint64_t, float>> hitrates_LFU;
	hitrates_LFU.reserve(N);

	auto cache_sizes_it = cache_sizes.begin();
	for (const auto& f : features)
	{
		Simulator<Policy::LFU, std::string> simulator;
		hitrates_LFU.push_back(simulator.hitrates(f, *(cache_sizes_it++)));
	}
	std::cout << chronometer.lap() << "s" << std::endl;


	/// OPT Policy
	std::cout << "OPT: calculating hitrate curves ... " << std::flush;

	std::vector<std::map<uint64_t, float>> hitrates_OPT;
	hitrates_OPT.reserve(N);

	cache_sizes_it = cache_sizes.begin();
	for (const auto& f : features)
	{
		Simulator<Policy::OPT, std::string> simulator;
		hitrates_OPT.push_back(simulator.hitrates(f, *(cache_sizes_it++)));
	}
	std::cout << chronometer.lap() << "s" << std::endl;

	std::cout << "Exporting to CSV files ... " << std::flush;
	export_csv(hitrates_LRU, hitrates_LFU, hitrates_OPT,
		"hitrates_f", selected_columns);
	export_perfprof("comparison.csv", hitrates_LRU, hitrates_LFU, hitrates_OPT);
	std::cout << chronometer.lap() << "s" << std::endl;

	std::cout << "Total time: " << chronometer.elapsed() << std::endl;

	return 0;
}

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param)
{
	if (argc == 1)
	{
		std::cout << "ERROR: specify input file" << std::endl;
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