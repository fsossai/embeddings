#ifndef __CACHED_SIM_HPP__
#define __CACHED_SIM_HPP__

#include <algorithm>
#include <cassert>
#include <ctime>
#include <memory>
#include <vector>
#include <unordered_map>

#include <fixed_size_heap.hpp>
#include "matrix.hpp"
#include "sharding.hpp"
#include "utils.hpp"

std::string get_timestamp(std::string format);

class Results
{
public:
    int P, D, N;
    Matrix<int> packets;
    Matrix<int> lookups;
    std::vector<int> fanout;
    std::vector<int> outgoing_packets;
    std::vector<int> outgoing_lookups;
    Matrix<int> packet_size;
    Matrix<int> outgoing_tables;
    Matrix<int> *cache_hits = nullptr;
    Matrix<int> *cache_refs = nullptr;
    Matrix<int> *cache_footprint = nullptr;
    int cache_aggregate_size;
    int cache_min_size;
    float cache_size_rel;
    std::vector<int> *cache_sizes = nullptr;
    std::string cache_policy = "";
    std::string cache_mode = "";
    std::string name = "";
    std::string sharding_mode = "";
    std::string sharding_file = "";
    std::string sharding_name = "";

    Results(int P, int D, int N);

    void print();

    void save(std::string output_directory);

private:
    const std::string _default_time_format = "%Y%m%d-%H%M%S";

    std::string get_default_name();
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

/// RESULTS

Results::Results(int P, int D, int N) :
    packets(Matrix<int>(P,P)),
    lookups(Matrix<int>(P,P)),
    fanout(std::vector<int>(P+1)),
    outgoing_packets(std::vector<int>(P)),
    outgoing_lookups(std::vector<int>(D+1)),
    packet_size(Matrix<int>(P, D+1)),
    outgoing_tables(Matrix<int>(P, D)),
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
    file << "\"cache_policy\" : " << '\"' << cache_policy << "\",\n";
    file << "\"cache_mode\" : " << '\"' << cache_mode << "\",\n";
    file << "\"sharding_mode\" : " << '\"' << sharding_mode << "\",\n";
    file << "\"sharding_file\" : " << '\"' << sharding_file << "\",\n";
    file << "\"sharding_name\" : " << '\"' << sharding_name << "\",\n";
    file << "\"packets\" : " << packets.to_string() << ",\n";
    file << "\"lookups\" : " << lookups.to_string() << ",\n";
    file << "\"outgoing_packets\" : " << vector_to_string(outgoing_packets) << ",\n";
    file << "\"outgoing_lookups\" : " << vector_to_string(outgoing_lookups) << ",\n";
    file << "\"outgoing_tables\" : " << outgoing_tables.to_string() << ", \n";
    file << "\"packet_size\" : " << packet_size.to_string() << ",\n";
    file << "\"fanout\" : " << vector_to_string(fanout);
    
    // cache hits
    if (cache_hits != nullptr)
    {
        file << ",\n";
        file << "\"cache_min_size\" : " << cache_min_size << ",\n";
        file << "\"cache_size_rel\" : " << cache_size_rel << ",\n";
        file << "\"aggregate_size\" : " << cache_aggregate_size << ",\n";
        file << "\"cache_sizes\" : " << vector_to_string(*cache_sizes) << ",\n";
        file << "\"cache_hits\" : " << cache_hits->to_string() << ",\n";
        file << "\"cache_refs\" : " << cache_refs->to_string();

        if (cache_footprint != nullptr)
        {
            file << ",\n";
            file << "\"cache_footprint\" : " << cache_footprint->to_string() << "\n";
        }
        else
            file << '\n';
    }
    else
        file << '\n';

    file << '}';
    file.flush();
    file.close();
}

std::string Results::get_default_name()
{
    return "sim_" + get_timestamp(_default_time_format);
}

struct pair_hash
{
    template<typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
    }
};


template<typename Tsharding, typename Tid,
    typename Tpolicy, typename Tmode, typename Tkey>
Results cached_simulation(
    const std::vector<std::vector<Tid>>& queries,
    int P,
    LookupProtocol<Tsharding, Tid>& protocol,
    Cache<Tpolicy, Tmode, Tkey>& cache)
{
    std::srand(std::time(0));
    const int N = queries.size();
    const int D = queries[0].size();

    Results results(P, D, N);
    results.cache_hits = &cache.hits;
    results.cache_refs = &cache.refs;
    results.cache_sizes = &cache.sizes;

    std::vector<int> lookups(D);

    for (const auto& query : queries)
    {
        // a random processor will take care of the query
        const int p = std::rand() % P;

        // resolving the query
        for (int i = 0; i < D; ++i)
        {
            // look up for 'query[i]' in table 'i'
            lookups[i] = protocol.lookup(i, query[i]);
            // checking cache of 'p' for each table
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
            results.packet_size.at(p, counts) += 1;
        }

        // keeping track of the tables that requests communications
        for (int i = 0; i < D; ++i)
        {
            if (lookups[i] != p)
                ++results.outgoing_tables.at(p, i);
        }
    }

    return results;
}


template<typename Tsharding, typename Tid>
Results noncached_simulation(
    const std::vector<std::vector<Tid>>& queries,
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

        // resolving the query
        for (int i = 0; i < D; ++i)
            lookups[i] = protocol.lookup(i, query[i]);

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
            results.packet_size.at(p, counts) += 1;
        }

        // keeping track of the tables that requests communications
        for (int i = 0; i < D; ++i)
        {
            if (lookups[i] != p)
                ++results.outgoing_tables.at(p, i);
        }
    }

    return results;
}


/*** Cache ***/

template<typename Tkey>
class Cache<Policy::LFU, Mode::Private, Tkey>
{
public:
    std::vector<int> sizes;
    int P, D;
    const std::string policy = "LFU";
    const std::string mode = "Private";
    int aggregate_size;
    Matrix<int> hits;
    Matrix<int> refs;

    Cache(std::vector<int> sizes, int P, int D) :
        sizes(sizes),
        P(P), D(D),
        _system(
            std::vector<std::vector<
                FixedSizeHeap<Tkey, uint64_t, std::less<uint64_t>>
            >>(P)
        ),
        hits(P, D),
        refs(P, D)
    {
        assert(sizes.size() == D);
        for (auto& p : _system)
        {
            p.reserve(D);
            for (auto size : sizes)
                p.push_back(FixedSizeHeap<Tkey, uint64_t, std::less<uint64_t>>(size));
        }
        aggregate_size = std::accumulate(sizes.begin(), sizes.end(), 0);
    }

    bool reference(int p, int table, Tkey id)
    {
        ++refs.at(p, table);
        if (_system[p][table].contains(id)) // cache hit
        {
            _system[p][table].change(id, [](auto val) { return val + 1; });
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


template<typename Tkey>
class Cache<Policy::LFU, Mode::Shared, Tkey>
{
public:
    std::vector<int> sizes;
    int P, D;
    const std::string policy = "LFU";
    const std::string mode = "Shared";
    int aggregate_size;
    Matrix<int> hits;
    Matrix<int> refs;

    Cache(int size, int P, int D) :
        sizes({size}),
        P(P), D(D),
        hits(P, D),
        refs(P, D)
    {
        _system.resize(P,
            FixedSizeHeap<std::pair<int, Tkey>, uint64_t,
                std::less<uint64_t>, pair_hash>(size));
        aggregate_size = size;
    }

    bool reference(int p, int table, Tkey id)
    {
        const std::pair<int, Tkey> key = std::make_pair(table, id);
        ++refs.at(p, table);
        if (_system[p].contains(key)) // cache hit
        {
            _system[p].change(key, [](auto val) { return val + 1; });
            ++hits.at(p, table);
            return true;
        }
        // cache miss
        _system[p].insert(key, 1);
        return false;
    }

    Matrix<int> get_tables_footprint()
    {
        Matrix<int> fp(P, D);
        fp.fill(0);
        for (int p = 0; p < P; ++p)
        {
            for (const auto& [key, val] : _system[p].get_raw_heap())
            {
                const auto& [table, id] = key;
                ++fp.at(p, table);
            }
        }
        return fp;
    }

private:
    std::vector<
        FixedSizeHeap<
            std::pair<int, Tkey>,
            uint64_t,
            std::less<uint64_t>,
            pair_hash
    >> _system;
};


#endif // #ifndef __CACHED_SIM_HPP__