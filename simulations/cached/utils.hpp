#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

char default_separator = ',';

template<typename T>
std::vector<std::vector<T>> parse_vector_of_dvectors(std::string filename);

template<typename T>
std::vector<std::vector<T>> parse_vector_of_fvectors(std::string filename);

template<typename T>
std::vector<T> parse_vector(std::string filename);

template<typename T, typename Converter>
std::vector<std::vector<T>>
_core_parse_vector_of_fvectors(std::string filename, Converter converter);


/*** DEFINITIONS ***/

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

template<>
std::vector<std::vector<std::string>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<std::string>(filename,
        [](const auto& s)
        { return std::move(s); }
    );
}

template<>
std::vector<std::vector<uint32_t>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<uint32_t>(filename,
        [](const auto& s)
        { return static_cast<uint32_t>(std::stoul(s)); }
    );
}

template<>
std::vector<std::vector<uint64_t>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<uint64_t>(filename,
        [](const auto& s)
        { return static_cast<uint64_t>(std::stoull(s)); }
    );
}

template<>
std::vector<std::vector<int>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<int>(filename,
        [](const auto& s)
        { return static_cast<int>(std::stoi(s)); }
    );
}

template<>
std::vector<std::vector<float>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<float>(filename,
        [](const auto& s)
        { return std::stof(s); }
    );
}

template<>
std::vector<std::vector<double>> parse_vector_of_fvectors(std::string filename)
{
    return _core_parse_vector_of_fvectors<double>(filename,
        [](const auto& s)
        { return std::stod(s); }
    );
}

template<typename T>
std::vector<T> parse_vector(std::string filename)
{
    return parse_vector_of_dvectors<T>(filename)[0];
}

template<typename T, typename Converter>
std::vector<std::vector<T>>
_core_parse_vector_of_fvectors(std::string filename, Converter converter)
{
    std::fstream file(filename);
    std::vector<std::vector<T>> data;
    std::string line;

    if (!std::getline(file, line))
        return std::vector<std::vector<T>>{};

    // parsing first line establishing the maximum size
    std::getline(file, line);
	const int max_size = std::count(
        line.begin(), line.end(), default_separator) + 1;
    file.seekg(0, std::ios_base::beg);

    while (std::getline(file, line))
    {
        size_t pos_start = 0, pos_end = 0, column = 0;
		std::vector<T> parsed_line;
		parsed_line.reserve(max_size);
		
		pos_end = line.find(default_separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
            parsed_line.push_back(
                converter(line.substr(pos_start, pos_end - pos_start)));
			pos_start = pos_end + 1;
			pos_end = line.find(default_separator, pos_start);
			++column;
		}
        parsed_line.push_back(
            converter(line.substr(pos_start, pos_end - pos_start)));
		data.push_back(std::move(parsed_line));
    }

    return data;
}

#endif // #ifndef __UTILS_HPP__