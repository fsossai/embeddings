#ifndef __CACHED_SIM_HPP__
#define __CACHED_SIM_HPP__

#include <algorithm>
#include <ctime>
#include <vector>
#include <unordered_set>

using matrix_t = std::vector<std::vector<float>>;

struct Results
{
    matrix_t packet;
    matrix_t lookup;
    matrix_t request;
    std::vector<int> fanout;
    float avg_fanout;
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

template<typename Tsharding, typename Tid>
Results cached_simulation(
    std::vector<std::vector<Tid>> queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol)
{
    std::srand(std::time(0));
    Results results;
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
        std::transform(
            lookups.begin(), lookups.end(), here.begin(),
            [p](const auto& l) { return p == l; }
        );

        // calculating fanout
        int fanout = std::unordered_set<int>(
            lookups.begin(), lookups.end()).size();

        cumulative_fanout += fanout;
        results.fanout.push_back(fanout);
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