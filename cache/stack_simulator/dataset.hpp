#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <string>

struct parser_parameters
{
	std::vector<int> selected_columns;
	int max_samples = std::numeric_limits<int>::max();
	char separator = '\t';
	std::string filename;
	int sparse_feat_offset = 0;
};


template<typename T>
class Dataset
{
protected:
	parser_parameters _param;
	bool _ready;
	bool check_index(const int index);
	bool check_file();
public:
	Dataset(const parser_parameters& param);
	Dataset(std::string filename);
	bool _all_cols = false;
};


template<typename T>
class RowMajorDataset : public Dataset<T>
{
private:
	std::vector<std::vector<T>> _samples;
public:
	using Dataset<T>::Dataset;
	std::vector<std::vector<T>> get_samples();
	bool import();
	void print();
};


template<typename T>
class ColMajorDataset : public Dataset<T>
{
private:
	std::vector<std::vector<T>> _features;
public:
	using Dataset<T>::Dataset;
	std::vector<std::vector<T>> get_features();
	bool import();
	void print();
};


/*** DEFINITIONS ***/

template<typename T>
Dataset<T>::Dataset(const parser_parameters& param)
    : _param(param),
    _ready(false)
{
	if (param.selected_columns.size() == 0)
		_all_cols = true;
}


template<typename T>
Dataset<T>::Dataset(std::string filename)
    : Dataset(parser_parameters())
{
	_param.filename = filename;
}


template<typename T>
bool Dataset<T>::check_index(const int index)
{
	if (_all_cols)
		return true;
	return std::find(
			_param.selected_columns.begin(),
			_param.selected_columns.end(),
			index
		) != _param.selected_columns.end();
}


template<typename T>
bool Dataset<T>::check_file()
{
	if (!std::filesystem::exists(_param.filename))
	{
		std::cerr << "File doesn't exists." << std::endl;
		return false;
	}
	return true;
}


/*** RowMajorDataset ***/

template<typename T>
std::vector<std::vector<T>>
RowMajorDataset<T>::get_samples()
{
	if (this->_ready)
        return _samples;
    throw std::logic_error("Dataset must be imported before using samples.");
}


template<>
bool RowMajorDataset<std::string>::import()
{
	int row_counter = 0;

	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;
    _samples.clear();

	if (!this->check_file())
		return false;

	while (row_counter < max && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0;
		std::vector<std::string> sample;
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				sample.push_back(std::move(
                    current_sample.substr(pos_start, pos_end - pos_start)
				));
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			sample.push_back(std::move(
                current_sample.substr(pos_start, pos_end - pos_start)
            ));
		}
		_samples.push_back(std::move(sample));
		row_counter++;
	}
    this->_ready = true;
	return true;
}


template<>
bool RowMajorDataset<uint32_t>::import()
{
	int row_counter = 0;

	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;
    _samples.clear();

	if (!this->check_file())
		return false;

	std::size_t sample_size = this->_param.selected_columns.size();
	if (sample_size == 0) // all columns have to be processed
	{
		// getting first sample in order to guess the number of columns
		std::getline(file, current_sample);
		sample_size = std::count(
			current_sample.begin(),
			current_sample.end(),
			this->_param.separator) + 1;
		file.seekg(0, std::ios_base::beg);
	}

	while (row_counter < max && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0;
		std::vector<uint32_t> sample;
		sample.reserve(sample_size);
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				sample.push_back(static_cast<uint32_t>(
					std::stoul(current_sample.substr(pos_start, pos_end - pos_start)
				)));
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			sample.push_back(static_cast<uint32_t>(
					std::stoul(current_sample.substr(pos_start, pos_end - pos_start)
			)));
		}
		_samples.push_back(std::move(sample));
		row_counter++;
	}
    this->_ready = true;
	return true;
}


template<typename T>
void RowMajorDataset<T>::print()
{
	int i = 0;
	for (const auto& sample : this->_samples)
	{
		std::printf("%i\t: ", i++);
		for (const auto& feature : sample)
			std::cout << feature << "\t";
		std::cout << std::endl;
	}
}


/*** ColMajorDataset ***/

template<typename T>
std::vector<std::vector<T>>
ColMajorDataset<T>::get_features()
{
	if (this->_ready)
        return _features;
    throw std::logic_error("Dataset must be imported before using samples.");
}


template<>
bool ColMajorDataset<std::string>::import()
{
	int row_counter = 0;
	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;

	if (!this->check_file())
		return false;

	for (auto& feature : _features)
		feature.clear();

    std::size_t sample_size = this->_param.selected_columns.size();
	if (sample_size == 0) // all columns have to be processed
	{
		// getting first sample in order to guess the number of columns
		std::getline(file, current_sample);
		sample_size = std::count(
			current_sample.begin(),
			current_sample.end(),
			this->_param.separator) + 1;
		file.seekg(0, std::ios_base::beg);
	}
	_features.resize(sample_size);

	while (row_counter < max && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0, sample_column = 0;
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				_features[sample_column++].push_back(
                    current_sample.substr(pos_start, pos_end - pos_start));
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			_features[sample_column].push_back(std::move(
                current_sample.substr(pos_start, pos_end - pos_start)));
		}
		row_counter++;
	}
    this->_ready = true;
	return true;
}


template<>
bool ColMajorDataset<int>::import()
{
	int row_counter = 0;
	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;

	if (!this->check_file())
		return false;

	for (auto& feature : _features)
		feature.clear();

    std::size_t sample_size = this->_param.selected_columns.size();
	if (sample_size == 0) // all columns have to be processed
	{
		// getting first sample in order to guess the number of columns
		std::getline(file, current_sample);
		sample_size = std::count(
			current_sample.begin(),
			current_sample.end(),
			this->_param.separator) + 1;
		file.seekg(0, std::ios_base::beg);
	}
	_features.resize(sample_size);

	while (row_counter < max && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0, sample_column = 0;
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				_features[sample_column++].push_back(std::stoi(
                    current_sample.substr(pos_start, pos_end - pos_start)));
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			_features[sample_column].push_back(std::stoi(
                current_sample.substr(pos_start, pos_end - pos_start)));
		}
		row_counter++;
	}
    this->_ready = true;
	return true;
}



template<>
bool ColMajorDataset<uint32_t>::import()
{
	int row_counter = 0;
	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;

	if (!this->check_file())
		return false;

	for (auto& feature : _features)
		feature.clear();

    std::size_t sample_size = this->_param.selected_columns.size();
	if (sample_size == 0) // all columns have to be processed
	{
		// getting first sample in order to guess the number of columns
		std::getline(file, current_sample);
		sample_size = std::count(
			current_sample.begin(),
			current_sample.end(),
			this->_param.separator) + 1;
		file.seekg(0, std::ios_base::beg);
	}
	_features.resize(sample_size);

	while (row_counter < max && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0, sample_column = 0;
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				_features[sample_column++].push_back(std::stoul(
                    current_sample.substr(pos_start, pos_end - pos_start)));
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			_features[sample_column].push_back(std::stoul(
                current_sample.substr(pos_start, pos_end - pos_start)));
		}
		row_counter++;
	}
    this->_ready = true;
	return true;
}


template<typename T>
void ColMajorDataset<T>::print()
{
	int i = 1;
	const int n_rows = _features[0].size();
	const int n_cols = _features.size();
	for (int i = 0; i < n_rows; i++)
	{
		std::printf("%i\t: ", i);
		for (int j = 0; j < n_cols; j++)
			std::cout << _features[j][i] << "\t";
		std::cout << std::endl;
	}
}

#endif // #ifndef __DATASET_HPP__