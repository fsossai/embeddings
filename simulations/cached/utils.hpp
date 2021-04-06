#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <sstream>
#include <vector>

template<typename T>
std::vector<std::vector<T>> parse_vector_of_dvectors(std::string filename);

template<typename T>
std::vector<std::vector<T>> parse_vector_of_fvectors(std::string filename);

template<typename T>
std::vector<T> parse_vector(std::string filename);


template<typename T>
std::vector<std::vector<T>> parse_vector_of_dvectors(std::string filename)
{
    std::fstream file(filename);
    std::vector<std::vector<T>> data;
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::vector<T> current;
        T t;
        while (ss >> t)
        {
            current.push_back(t);
            ss.ignore(1);
        }
        data.push_back(std::move(current));
    }

    return data;
}

template<typename T>
std::vector<std::vector<T>> parse_vector_of_fvectors(std::string filename)
{
    std::fstream file(filename);
    std::vector<std::vector<T>> data;
    std::string line;

    if (!std::getline(file, line))
        return std::vector<std::vector<T>>{};

    // parsing first line establishing the size
    std::stringstream ss(line);
    std::vector<T> first;
    T t;
    while (ss >> t)
    {
        first.push_back(t);
        ss.ignore(1);
    }
    data.push_back(first);
    const int max_size = first.size();

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::vector<T> current(max_size);
        T t;
        for (int i = 0; i < max_size && ss >> t; ++i)
        {
            current[i] = t;
            ss.ignore(1);
        }
        data.push_back(std::move(current));
    }

    return data;
}

template<typename T>
std::vector<T> parse_vector(std::string filename)
{
    return parse_vector_of_dvectors<T>(filename)[0];
}


#endif // #ifndef __UTILS_HPP__