#include <iostream>
#include <vector>
#include <numeric>

#include <chronometer.hpp>
#include <cxxopts.hpp>
#include "cached_sim.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
	cxxopts::Options options("sim", "Cached simulator");
	options.add_options()
		("q,queries", "Input query file", cxxopts::value<std::string>())
		("c,counts", "Feature cardinalities", cxxopts::value<std::string>())
		("m,min-size", "Cache minumum size for each processor", cxxopts::value<int>()->default_value("100"))
		("r,rel-size","Fraction of cardinality to be used as cache", cxxopts::value<float>()->default_value("0.01"))
		("p,n-processors", "List of processors to be used", cxxopts::value<std::vector<int>>())
		("l,lookup-table", "Specify a precompiled lookup table", cxxopts::value<std::string>())
		("n,sharding-name", "Technical name of the sharding strategy", cxxopts::value<std::string>())
	;
	
	auto args = options.parse(argc, argv);
	if (args.count("help"))
	{
		std::cout << options.help() << std::endl;
		return 0;
	}

	// Checking arguments
	auto Ps = args["n-processors"].as<std::vector<int>>();
	if (Ps.size() == 0)
	{
		std::cerr << "Please specify the number of processors to proceed with (--n-processors)";
		return -1;
	}

    Chronometer chronometer;
    std::cout << "Reading dataset ... " << std::flush;
    chronometer.start();
	auto queries = parse_vector_of_fvectors<uint32_t>(
		args["queries"].as<std::string>()
	);
	std::cout << chronometer.lap() << "s" << std::endl;

	const int D = queries[0].size();
	const int N = queries.size();
	const int min_size = args["min-size"].as<int>();
	const float size_relative = args["rel-size"].as<float>();

	auto counts = parse_vector<int>(args["counts"].as<std::string>());
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
	for (int P : Ps)
	{
    	//LookupProtocol<Sharding::Random, uint32_t> protocol(P);
    	LookupProtocol<Sharding::Custom, uint32_t> protocol(
			args["lookup-table"].as<std::string>()
		);
		//LookupProtocol<Sharding::Hybrid, uint32_t> protocol(
		//	P, args["lookup-table"].as<std::string>()
		//);
		//Cache<Policy::LFU, Mode::Private, uint32_t> cache(sizes, P, D);
		Cache<Policy::LFU, Mode::Shared, uint32_t> cache(aggregate_size, P, D);

		std::cout << "P = " << P << ", ";
		std::cout << "Protocol: " << protocol.name << ", ";
		std::cout << "Cache: " << cache.policy << " " << cache.mode << " ... ";

		//Results results = noncached_simulation(queries, P, protocol);
		///*
		Results results = cached_simulation(queries, P, protocol, cache);
		results.cache_sizes = &sizes;
		results.cache_min_size = min_size;
		results.cache_size_rel = size_relative;
		results.cache_policy = cache.policy;
    	results.cache_aggregate_size = aggregate_size;
    	results.cache_mode = cache.mode;
		auto footprint = cache.get_tables_footprint();
		results.cache_footprint = &footprint;
		//*/
    	results.sharding_mode = protocol.name;
		results.sharding_file = args["lookup-table"].as<std::string>();
		if (args.count("sharding-name"))
			results.sharding_name = args["sharding-name"].as<std::string>();
		results.save("results");
		std::cout << chronometer.lap() << "s" << std::endl;
	}

    std::cout << "Total time: " << chronometer.elapsed() << std::endl;

    return 0;
}
