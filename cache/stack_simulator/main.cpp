#include <iostream>
#include <vector>

#include "chronometer.hpp"
#include "dataset.hpp"
#include "simulator.hpp"

bool set_cmdline_args(int argc, char **argv, parser_parameters_t& param);
std::vector<uint64_t> get_sizes(
	std::map<std::pair<uint64_t, float>, float> hitrates);
void export_csv(
    const std::vector<std::map<std::pair<uint64_t, float>, float>>& hitrates_LRU,
    const std::vector<std::map<uint64_t, float>>& hitrates_LFU,
    const std::string& basename,
    const int feature_offset);

int main(int argc, char** argv)
{
	using namespace cache;

	Chronometer chronometer;
	const int N = 3;
	const int SPARSE_OFFSET = 14;
	const std::vector<float> cache_sizes_percentage{0.05, 0.1, 0.15};

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
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "LFU: calculating hitrate curves ... " << std::flush;

	std::vector<std::map<uint64_t, float>> hitrates_LFU;
	hitrates_LFU.reserve(N);
	auto cache_sizes_it = cache_sizes.begin();
	for (auto v : cache_sizes) //D
	{
		for (auto s : v)
			std::cout << s << ',';
		std::cout << '\n';
	}

	for (const auto& f : features)
	{
		Simulator<Policy::LFU, std::string> simulator;
		hitrates_LFU.push_back(simulator.hitrates(f, *(cache_sizes_it++)));
	}
	std::cout << chronometer.lap() << "s" << endl;

	std::cout << "Exporting to CSV files ... " << std::flush;
	export_csv(hitrates_LRU, hitrates_LFU, "hitrates_f", SPARSE_OFFSET);
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

std::vector<uint64_t> get_sizes(
	std::map<std::pair<uint64_t, float>, float> hitrates)
{
	std::vector<uint64_t> sizes(hitrates.size());

	std::transform(
		hitrates.begin(),
		hitrates.end(),
		sizes.begin(),
		[](const auto& p) { return p.first.first; }
	);

	return sizes;
}

void export_csv(
    const std::vector<std::map<std::pair<uint64_t, float>, float>>& hitrates_LRU,
    const std::vector<std::map<uint64_t, float>>& hitrates_LFU,
    const std::string& basename,
    const int feature_offset)
{
	const int N = hitrates_LRU.size();

	for (int i = 0; i<N; i++)
	{
		std::string name =
            basename + std::to_string(feature_offset + i) + ".csv";
		std::ofstream file(name, std::ofstream::binary);
		file.precision(5);
		file << "cache_size,cache_size_relative,hitrate_LRU,hitrate_LFU\n";

		auto LFU_it = hitrates_LFU[i].begin();
		for (const auto& [key, h_LRU] : hitrates_LRU[i])
		{
			const auto [cs, cs_rel] = key;
			const auto [_, h_LFU] = *(LFU_it++);
			file << cs << ',' << cs_rel << ',' << h_LRU << ',' << h_LFU << '\n';
		}
		
		file << endl;
		file.flush();
		file.close();
	}
}