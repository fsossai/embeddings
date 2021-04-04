#ifndef __CACHED_SIM_HPP__
#define __CACHED_SIM_HPP__

#include <algorithm>
#include <cassert>
#include <ctime>
#include <vector>
#include <unordered_map>

#include <fixed_size_heap.hpp>
#include "matrix.hpp"

std::string get_timestamp(std::string format);

template<typename T>
std::string vector_to_string(std::vector<T> v);

class Results
{
public:
    int P, D, N;
    Matrix<int> packets;
    Matrix<int> lookups;
    std::vector<int> fanout;
    std::vector<int> outgoing_packets;
    std::vector<int> outgoing_lookups;
    std::string name = "";

    Results(int P, int D, int N);

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
};

class Policy
{
public:
    class LRU;
    class LFU;
};

class Mode
{
public:
    class Shared;
    class Private;
};


template<typename Tsharding, typename Tid>
class LookupProtocol;

template<typename Tpolicy, typename Tmode, typename Tkey>
class Cache;


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


template<typename T>
std::string vector_to_string(std::vector<T> v)
{
    std::stringstream ss("");
    ss << '[';

    for (int i = 0; i < v.size() - 1; ++i)
        ss << v[i] << ',';
    
    ss << v[v.size() - 1] << ']';

    return ss.str();
}

/// RESULTS

Results::Results(int P, int D, int N) :
    packets(Matrix<int>(P,P)),
    lookups(Matrix<int>(P,P)),
    fanout(std::vector<int>(P+1)),
    outgoing_packets(std::vector<int>(D+1)),
    outgoing_lookups(std::vector<int>(D+1)),
    P(P), D(D), N(N)
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
    // creating a JSON file

    // setting a default name if not specified
    if (name.length() == 0)
        name = get_default_name();
    
    std::ofstream file(
        output_directory + "\\" + name + ".json",
        std::ios_base::binary);

    file << "{\n";
    file << "\"processors\" : " << P << ",\n";
    file << "\"tables\" : " << D << ",\n";
    file << "\"queries\" : " << N << ",\n";
    file << "\"packets\" : " << packets.to_string() << ",\n";
    file << "\"lookups\" : " << lookups.to_string() << ",\n";
    file << "\"outgoing_packets\" : " <<
        vector_to_string(outgoing_packets) << ",\n";
    file << "\"outgoing_lookups\" : " <<
        vector_to_string(outgoing_lookups) << ",\n";
    file << "\"fanout\" : " <<
        vector_to_string(fanout) << '\n';
    file << '}';
    file.flush();
    file.close();
}

std::string Results::get_default_name()
{
    return "sim_" + get_timestamp(_default_time_format);
}


template<typename Tsharding, typename Tid,
    typename Tpolicy, typename Tmode, typename Tkey>
Results cached_simulation(
    std::vector<std::vector<Tid>> queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol,
    Cache<Tpolicy, Tmode, Tkey> cache)
{
    std::srand(std::time(0));
    const int N = queries.size();
    const int D = queries[0].size();

    Results results(P, D, N);

    std::vector<int> lookups(D);

    for (const auto& query : queries)
    {
        // a random processor will take care of the query
        const int p = std::rand() % P;

        // look up for every id in the query
        std::transform(query.begin(), query.end(), lookups.begin(),
            [&](int id) { return protocol.lookup(id); });

        // checking cache of 'p' for each table
        for (int i = 0; i < D; ++i)
        {
            if (lookups[i] != p && cache.reference(p, i, query[i]))
                lookups[i] = p; // no outgoing lookup or packet
        }

        // counting the number of distinct lookups
        std::unordered_map<int, int> lcounts;
        for (auto& l : lookups)
            ++lcounts[l];

        // calculating fanout histogram
        ++results.fanout[lcounts.size()];

        // identifying outgoing data
        int local_lookups = lcounts[p];
        lcounts.erase(p);
        ++results.outgoing_packets[lcounts.size()];
        ++results.outgoing_lookups[D - local_lookups];

        // accounting for all sent packets
        for (const auto& [i, counts] : lcounts)
        {
            results.packets.at(p, i) += 1;
            results.lookups.at(p, i) += counts;
        }
    }

    return results;
}


template<typename Tsharding, typename Tid>
Results noncached_simulation(
    std::vector<std::vector<Tid>> queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol)
{
    std::srand(std::time(0));
    const int N = queries.size();
    const int D = queries[0].size();

    Results results(P, D, N);

    std::vector<int> lookups(D);

    for (const auto& query : queries)
    {
        // a random processor will take care of the query
        const int p = std::rand() % P;

        // look up for every id in the query
        std::transform(query.begin(), query.end(), lookups.begin(),
            [&](auto id) { return protocol.lookup(id); });

        // counting the number of distinct lookups
        std::unordered_map<int, int> lcounts;
        for (auto& l : lookups)
            ++lcounts[l];

        // calculating fanout histogram
        ++results.fanout[lcounts.size()];

        // identifying outgoing data
        int local_lookups = lcounts[p];
        lcounts.erase(p);
        ++results.outgoing_packets[lcounts.size()];
        ++results.outgoing_lookups[D - local_lookups];

        // accounting for all sent packets
        for (const auto& [i, counts] : lcounts)
        {
            results.packets.at(p, i) += 1;
            results.lookups.at(p, i) += counts;
        }
    }

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


template<typename Tkey>
class Cache<Policy::LFU, Mode::Private, Tkey>
{
public:
    std::vector<int> sizes;
    int P, D;
    const std::string name = "LFU,Private";
    Matrix<int> hits;

    Cache(std::vector<int> sizes, int P, int D) :
        sizes(sizes),
        P(P), D(D),
        _system(
            std::vector<std::vector<
                FixedSizeHeap<Tkey, uint64_t, std::less<uint64_t>>
            >>(P)
        ),
        hits(P, D)
    {
        assert(sizes.size() == D);
        for (auto& p : _system)
        {
            p.reserve(D);
            for (auto size : sizes)
                p.push_back(FixedSizeHeap<Tkey, uint64_t, std::less<uint64_t>>(size));
        }
    }

    bool reference(int p, int table, Tkey id)
    {
        if (_system[p][table].contains(id)) // cache hit
        {
            FixedSizeHeap<int,int,std::less<int>> f;
            _system[p][table].change(id, [](uint64_t val) { return val + 1; });
            //_system[p][table].set(id, _system[p][table].get(id) + 1);
            ++hits.at(p, table);
            return true;
        }
        // cache miss
        _system[p][table].insert(id, 1);
        return false;
    }

private:
    std::vector<std::vector<
        FixedSizeHeap<Tkey, uint64_t, std::less<uint64_t>>
    >> _system;
};


#endif // #ifndef __CACHED_SIM_HPP__