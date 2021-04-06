#ifndef __SHARDING_HPP__
#define __SHARDING_HPP__

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

//#include <dataset.hpp>

class Sharding
{
public:
    class Random;
    class Custom;
};

template<typename T>
struct CustomPair
{
    T key;
    int val;

    friend std::istream& operator>>(std::istream& s, CustomPair& p)
    {
        s >> p.key;
        s.ignore(1);
        s >> p.val;
        return s;
    }
};

template<typename Tsharding, typename Tid>
class LookupProtocol;

template<typename T>
std::vector<std::unordered_map<T, int>>
read_lookup_table(std::string filename);

template<typename Tsharding>
void create_lookup_table(std::string input_file, std::string output_file, int P);


/*** LookupProtocol ***/

template<>
class LookupProtocol<Sharding::Random, uint32_t>
{
public:
    LookupProtocol(int P) : P(P)
    { }

    int lookup(int table, uint32_t id) const
    {
        return id % P;
    }

    const int P;
    const std::string name = "Random";
};


template<>
class LookupProtocol<Sharding::Custom, uint32_t>
{
public:
    LookupProtocol(std::string lookup_table_file) :
        _lookup_table_file(lookup_table_file)
    {
        _ltable = read_lookup_table<uint32_t>(lookup_table_file);
    }

    int lookup(int table, uint32_t id) const
    {
        return _ltable[table].at(id);
    }

    const std::string name = "X";

private:
    const std::string _lookup_table_file;
    std::vector<std::unordered_map<uint32_t, int>> _ltable;
};


/*** Lookup Table utilities ***/

template<typename T>
std::vector<std::unordered_map<T, int>>
read_lookup_table(std::string filename)
{
    std::fstream file(filename);
    std::string line;
    std::vector<std::unordered_map<T, int>> ltable;

    // each line represents an embedding table
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::unordered_map<T, int> m;
        CustomPair<uint32_t> p;
        while (ss >> p)
        {
            m[p.key] = p.val;
            ss.ignore(1);
        }
        ltable.push_back(m);
    }
    return ltable;
}


template<>
void create_lookup_table<Sharding::Random>(
    std::string input_file, std::string output_file, int P)
{
    parser_parameters param;
    param.filename = input_file;
    param.separator = ',';
    ColMajorDataset<uint32_t> ds(param);
    ds.import();
    auto features = ds.get_features();
    const int D = features.size();

    std::ofstream output(output_file, std::ios_base::binary);

    for (const auto& f : features)
    {
        std::unordered_set<uint32_t> emb(f.begin(), f.end());

        auto it = emb.begin();
        output << *(it++) << ':' << (rand() % P);
        while (it != emb.end())
            output << ',' << *(it++) << ':' << (rand() % P);

        output << '\n';
    }
}

#endif // #ifndef __SHARDING_HPP__