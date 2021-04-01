#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <vector>

std::vector<uint64_t> get_sizes(
	std::map<std::pair<uint64_t, float>, float> hitrates);
void export_csv(
    const std::vector<std::map<std::pair<uint64_t, float>, float>>& hitrates_LRU,
    const std::vector<std::map<uint64_t, float>>& hitrates_LFU,
    const std::vector<std::map<uint64_t, float>>& hitrates_OPT,
    const std::string& basename,
    const std::vector<int>& selected_columns);
void export_perfprof(
    const std::string& output_name,
    const std::vector<std::map<std::pair<uint64_t, float>, float>>& hitrates_LRU,
    const std::vector<std::map<uint64_t, float>>& hitrates_LFU,
    const std::vector<std::map<uint64_t, float>>& hitrates_OPT);

constexpr int OUTPUT_FP_PRECISION = 5;

/*** DEFINITIONS ***/

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
    const std::vector<std::map<uint64_t, float>>& hitrates_OPT,
    const std::string& basename,
    const std::vector<int>& selected_columns)
{
	const int N = hitrates_LRU.size();
	auto column_it = selected_columns.begin();
	for (int i = 0; i<N; i++)
	{
		std::string name =
            basename + std::to_string(*(column_it++)) + ".csv";
		std::ofstream file(name, std::ofstream::binary);
		file.precision(OUTPUT_FP_PRECISION);
		file << "cache_size,cache_size_relative,";
		file << "hitrate_LRU,hitrate_LFU,hitrate_OPT\n";

		auto LFU_it = hitrates_LFU[i].begin();
		auto OPT_it = hitrates_OPT[i].begin();
		for (const auto& [key, h_LRU] : hitrates_LRU[i])
		{
			const auto [cs, cs_rel] = key;
			const auto [_ignore1, h_LFU] = *(LFU_it++);
			const auto [_ignore2, h_OPT] = *(OPT_it++);
			file << cs << ',' << cs_rel << ',';
			file << h_LRU << ',' << h_LFU << ',' << h_OPT << '\n';
		}
		
		file << endl;
		file.flush();
		file.close();
	}
}

void export_perfprof(
    const std::string& output_name,
    const std::vector<std::map<std::pair<uint64_t, float>, float>>& hitrates_LRU,
    const std::vector<std::map<uint64_t, float>>& hitrates_LFU,
    const std::vector<std::map<uint64_t, float>>& hitrates_OPT)
{
	const int N = hitrates_LRU.size();
	std::ofstream file(output_name, std::ofstream::binary);
	file.precision(OUTPUT_FP_PRECISION);
	file << "LRU,LFU,OPT\n";

	for (int i = 0; i<N; i++)
	{
		auto LFU_it = hitrates_LFU[i].begin();
		auto OPT_it = hitrates_OPT[i].begin();
		for (const auto& [key, h_LRU] : hitrates_LRU[i])
		{
			const auto [_ignore1, h_LFU] = *(LFU_it++);
			const auto [_ignore2, h_OPT] = *(OPT_it++);
			file << h_LRU << ',' << h_LFU << ',' << h_OPT << '\n';
		}
		
	}
	file.close();
}

#endif // #ifndef __UTILITIES_HPP__
