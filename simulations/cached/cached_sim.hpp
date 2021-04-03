#ifndef __CACHED_SIM_HPP__
#define __CACHED_SIM_HPP__

#include <algorithm>
#include <ctime>
#include <vector>
#include <unordered_map>

#include "matrix.hpp"

std::string get_timestamp(std::string format);

class Results
{
public:
    int P;
    Matrix<int> packets;
    Matrix<int> lookups;
    std::vector<int> fanout;
    std::vector<int> outgoing_packets;
    std::vector<int> outgoing_lookups;
    float avg_fanout;
    std::string name = "";

    Results(int P);

    void print();

    void save(std::string output_directory);

private:
    const std::string _default_time_format = "%Y%m%d-%H%M%S";

    std::string get_default_name();
};


class Sharding
{
public:
    class Random;
    class X;
};


template<typename Tsharding, typename Tid>
class LookupProtocol;


/*** DEFINITIONS ***/

std::string get_timestamp(std::string format)
{
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), format.c_str(), timeinfo);
    std::string str(buffer);

    return str;
}

/// RESULTS

Results::Results(int P) :
    packets(Matrix<int>(P,P)),
    lookups(Matrix<int>(P,P)),
    P(P)
{ }

void Results::print()
{
    std::cout << "--- Simulation with P = " << P << '\n';
    std::cout << " Packet Matrix:\n";
    packets.print(std::cout);
    std::cout << " Lookups Matrix:\n";
    lookups.print(std::cout);
}

void Results::save(std::string output_directory)
{
    if (name.length() == 0)
        name = get_default_name();

    std::ofstream file_packets(
        output_directory + "\\" + name + "_packets.txt",
		std::ios_base::binary);

    std::ofstream file_lookups(
        output_directory + "\\" + name + "_lookups.txt",
		std::ios_base::binary);

    std::ofstream file_outgoing(
        output_directory + "\\" + name + "_outgoing.csv",
        std::ios_base::binary);

    packets.print(file_packets);
    lookups.print(file_lookups);

    // checking sizes consistency
    if (outgoing_packets.size() != outgoing_lookups.size() ||
        outgoing_lookups.size() != fanout.size())
    {
        throw std::logic_error("Results::save: Inconsistent sizes");
    }

    // printing header of CSV file
    file_outgoing << "fanout,outgoing_packets,outgoing_lookups\n";
    
    const int N = fanout.size();
    for (int i = 0; i < N; ++i)
    {
        file_outgoing << fanout[i] << ',';
        file_outgoing << outgoing_packets[i] << ',';
        file_outgoing << outgoing_lookups[i] << '\n';
    }
}

std::string Results::get_default_name()
{
    return "sim_" + get_timestamp(_default_time_format);
}


template<typename Tsharding, typename Tid>
Results cached_simulation(
    std::vector<std::vector<Tid>> queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol)
{
    std::srand(std::time(0));
    const int N = queries.size();
    const int D = queries[0].size();

    Results results(P);
    results.fanout.reserve(N);
    results.outgoing_packets.reserve(N);
    results.outgoing_lookups.reserve(N);

    std::vector<int> lookups(D);
    float cumulative_fanout = 0.0f;

    for (const auto& query : queries)
    {
        // a random processor will take care of the query
        const int p = std::rand() % P;

        // look up for every id in the query
        std::transform(query.begin(), query.end(), lookups.begin(),
            [&](int id) { return protocol.lookup(id); }
        );

        // counting the number of distinct lookups
        std::unordered_map<int, int> lcounts;
        std::for_each(
            lookups.begin(), lookups.end(),
            [&](int l) { ++lcounts[l]; }
        );

        // calculating fanout
        int fanout = lcounts.size();
        cumulative_fanout += fanout;
        results.fanout.push_back(fanout);

        // identifying outgoing data
        int local_lookups = lcounts[p];
        lcounts.erase(p);
        results.outgoing_packets.push_back(lcounts.size());
        results.outgoing_lookups.push_back(D - local_lookups);

        // accounting for all sent packets
        for (const auto& [i, counts] : lcounts)
        {
            results.packets.at(p, i) += 1;
            results.lookups.at(p, i) += counts;
        }
    }

    results.avg_fanout = cumulative_fanout / queries.size();
    return results;
}

template<>
class LookupProtocol<Sharding::Random, uint32_t>
{
public:
    LookupProtocol(int n_processors) : P(n_processors)
    { }

    int lookup(uint32_t id)
    {
        return id % P;
    }

    const int P;
};

template<>
class LookupProtocol<Sharding::X, uint32_t>
{
public:
    LookupProtocol() = default;

    int lookup(uint32_t id) { return 0; }
};


#endif // #ifndef __CACHED_SIM_HPP__