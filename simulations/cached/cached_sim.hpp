#ifndef __CACHED_SIM_HPP__
#define __CACHED_SIM_HPP__

#include <algorithm>
#include <ctime>
#include <vector>
#include <unordered_map>

#include "matrix.hpp"

struct Results
{
    int P;
    Matrix<int> packets;
    Matrix<int> lookups;
    std::vector<int> fanout;
    float avg_fanout;

    Results(int P) :
        packets(Matrix<int>(P,P)),
        lookups(Matrix<int>(P,P)),
        P(P)
    { }

    void print()
    {
        std::cout << "--- Simulation with P = " << P << '\n';
        std::cout << " Packet Matrix:\n";
        packets.print(std::cout);
        std::cout << " Lookups Matrix:\n";
        lookups.print(std::cout);
    }
};


class Sharding
{
public:
    class Random;
    class X;
};


template<typename Tsharding, typename Tid>
class LookupProtocol;

template<typename Tsharding, typename Tid>
Results cached_simulation(
    std::vector<std::vector<Tid>> queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol)
{
    std::srand(std::time(0));
    Results results(P);
    const int D = queries[0].size();
    std::vector<int> lookups(D);
    std::vector<bool> here(D);
    float cumulative_fanout = 0.0f;

    for (const auto& query : queries)
    {
        // a random processor will take care of the query
        const int p = std::rand() % P;

        // look up for every id in the query
        std::transform(query.begin(), query.end(), lookups.begin(),
            [&](int id) { return protocol.lookup(id); }
        );

        // finding which id are in the memory of proc. 'p'
        /*std::transform(
            lookups.begin(), lookups.end(), here.begin(),
            [p](const auto& l) { return p == l; }
        );*/

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

        // accounting for all sent packets
        lcounts.erase(p);
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


void print_sim_results(const Results& results)
{
    
}

#endif // #ifndef __CACHED_SIM_HPP__