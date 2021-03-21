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

	std::cout << "Calculating hitrate curves ... " << std::flush;
	std::vector<std::map<uint64_t, float>> hitrates;
	hitrates.reserve(N);
	for (const auto& f : features)
	{
		Simulator<Policy::LRU, std::string> simulator;
		hitrates.push_back(simulator.hitrates(f));
	}
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Exporting to CSV files ... " << std::flush;
	export_csv(hitrates, "hitrates_F", SPARSE_OFFSET);
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