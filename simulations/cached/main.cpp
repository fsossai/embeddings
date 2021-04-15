#include <iostream>
#include <vector>
#include <numeric>

#include <chronometer.hpp>

#include "cached_sim.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "ERROR: expecting 2 input arguments\n";
		return -1;
	}

    Chronometer chronometer;
    std::cout << "Reading dataset ... " << std::flush;
    chronometer.start();
	auto queries = parse_vector_of_fvectors<uint32_t>(std::string(argv[1]));
	std::cout << chronometer.lap() << "s" << std::endl;

	const int D = queries[0].size();
	const int N = queries.size();
	const int min_size = 100;
	const float size_relative = 0.01;

	auto counts = parse_vector<int>(std::string(argv[2]));
	std::vector<int> sizes(D);
	std::transform(
		counts.begin(), counts.end(), sizes.begin(),
		[=](int s)
		{
			auto fraction = s * size_relative;
			if (s <= min_size)
				return s;
			if (fraction < min_size)
				return min_size;
			return static_cast<int>(fraction);
		}
	);
	int aggregate_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    /// Starting simulations
	for (int P : {16})
	{
    	LookupProtocol<Sharding::Random, uint32_t> protocol(P);
		//Cache<Policy::LFU, Mode::Private, uint32_t> cache(sizes, P, D);
		Cache<Policy::LFU, Mode::Shared, uint32_t> cache(aggregate_size, P, D);

		std::cout << "P = " << P << ", ";
		std::cout << "Protocol: " << protocol.name << ", ";
		std::cout << "Cache: " << cache.policy << " " << cache.mode << " ... ";

		//Results results = noncached_simulation(queries, P, protocol);
		Results results = cached_simulation(queries, P, protocol, cache);
		results.cache_sizes = sizes;
		results.cache_min_size = min_size;
		results.cache_size_rel = size_relative;
		auto footprint = cache.get_tables_footprint();
		results.cache_footprint = &footprint;
		results.save("results");
		std::cout << chronometer.lap() << "s" << std::endl;
	}

    std::cout << "Total time: " << chronometer.elapsed() << std::endl;

    return 0;
}
