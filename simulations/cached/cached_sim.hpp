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

std::string get_timestamp(std::string format);

template<typename T>
std::string vector_to_string(std::vector<T>& v);

class Results
{
public:
    int P, D, N;
    Matrix<int> packets;
    Matrix<int> lookups;
    std::vector<int> fanout;
    std::vector<int> outgoing_packets;
    std::vector<int> outgoing_lookups;
    Matrix<int> *cache_hits = nullptr;
    Matrix<int> *cache_refs = nullptr;
    std::vector<int> *cache_sizes = nullptr;
    std::string cache_policy = "";
    std::string cache_mode = "";
    std::string name = "";
    std::string sharding = "";

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


template<typename T>
std::string vector_to_string(std::vector<T>& v)
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
    outgoing_packets(std::vector<int>(P+1)),
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
    file << "\"cache_policy\" : " << '\"' << cache_policy << "\",\n";
    file << "\"cache_mode\" : " << '\"' << cache_mode << "\",\n";
    file << "\"sharding\" : " << '\"' << sharding << "\",\n";
    file << "\"packets\" : " << packets.to_string() << ",\n";
    file << "\"lookups\" : " << lookups.to_string() << ",\n";
    file << "\"outgoing_packets\" : " <<
        vector_to_string(outgoing_packets) << ",\n";
    file << "\"outgoing_lookups\" : " <<
        vector_to_string(outgoing_lookups) << ",\n";
    file << "\"fanout\" : " <<
        vector_to_string(fanout);
    
    // cache hits
    if (cache_hits != nullptr)
    {
        file << ",\n";
        file << "\"cache_sizes\" : " << vector_to_string(*cache_sizes) << ",\n";
        file << "\"cache_hits\" : " << cache_hits->to_string() << ",\n";
        file << "\"cache_refs\" : " << cache_refs->to_string() << "\n";
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
    results.cache_policy = cache.policy;
    results.cache_mode = cache.mode;
    results.sharding = protocol.name;

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
    results.sharding = protocol.name;

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
    }

    bool reference(int p, int table, Tkey id)
    {
        const std::pair<int, int> key = std::make_pair(table, id);
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