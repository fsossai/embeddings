#include <iostream>
#include <vector>

#include "chronometer.hpp"
#include "dataset.hpp"
#include "simulator.hpp"

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param);

int main(int argc, char** argv)
{
	using namespace cache;

	Chronometer chronometer;
	const int N = 26;
	const int SPARSE_OFFSET = 14;

	std::vector<int> selected_columns(N);
	std::iota(selected_columns.begin(), selected_columns.end(), SPARSE_OFFSET);
	
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
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "LRU: calculating hitrate curves ... " << std::flush;
	std::vector<std::map<uint64_t, float>> hitrates_LRU;
	hitrates_LRU.reserve(N);
	for (const auto& f : features)
	{
		Simulator<Policy::LRU, std::string> simulator;
		hitrates_LRU.push_back(simulator.hitrates(f));
	}
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "LFU: calculating hitrate curves ... " << std::flush;
	std::vector<std::map<uint64_t, float>> hitrates_LFU;
	hitrates_LFU.reserve(N);
	for (const auto& f : features)
	{
		Simulator<Policy::LFU, std::string> simulator;
		hitrates_LFU.push_back(simulator.hitrates(f, {10,100,1000}));
	}
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Exporting to CSV files ... " << std::flush;
	//export_csv(hitrates_LRU, "hitrates_F", SPARSE_OFFSET);
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