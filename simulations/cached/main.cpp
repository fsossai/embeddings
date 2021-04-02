#include <iostream>
#include <vector>

#include <dataset.hpp>
#include <chronometer.hpp>

#include "cached_sim.hpp"

bool set_cmdline_args(int argc, char **argv, parser_parameters& param);

int main(int argc, char **argv)
{
    Chronometer chronometer;

	parser_parameters param;
    param.separator = ',';

    if (!set_cmdline_args(argc, argv, param))
		return -1;

    std::cout << "Reading dataset ... " << std::flush;
    chronometer.start();
	RowMajorDataset<int> dataset(param);
	dataset.import();
	auto queries = dataset.get_samples();
	std::cout << chronometer.lap() << "s" << std::endl;

    /// Starting simulation
    chronometer.start();
    int P = 1;
    LookupProtocol<Sharding::Random, int> protocol(P);
    Results results = cached_simulation(queries, P, protocol);
    std::cout << "Fanout: ";
    for (auto f : results.fanout)
    {
        std::cout << f << ' ';
    }
    std::cout << '\n';
	std::cout << chronometer.lap() << "s" << std::endl;


    std::cout << "Total time: " << chronometer.elapsed() << std::endl;

    return 0;
}

bool set_cmdline_args(int argc, char **argv, parser_parameters& param)
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