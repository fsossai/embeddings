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

typedef struct parser_parameters
{
	std::vector<int> selected_columns;
	int max_samples;
	char separator;
	std::string filename;
	int sparse_feat_offset;
} parser_parameters_t;


template<std::size_t N>
class Dataset
{
protected:
	parser_parameters_t _param;
	bool _ready;
	bool check_index(const int index);
	bool check_file();
public:
	Dataset(const parser_parameters_t& param);
};


template<std::size_t N>
class RowMajorDataset : public Dataset<N>
{
private:
	std::vector<std::array<std::string, N>> _samples;
public:
	RowMajorDataset(const parser_parameters_t& param);
	std::vector<std::array<std::string, N>> get_samples();
	bool import();
	void print();
};


template<std::size_t N>
class ColMajorDataset : public Dataset<N>
{
private:
	std::array<std::vector<std::string>, N> _features;
public:
	ColMajorDataset(const parser_parameters_t& param);
	std::array<std::vector<std::string>, N> get_features();
	bool import();
	void print();
};


/*** DEFINITIONS ***/


template<std::size_t N>
Dataset<N>::Dataset(const parser_parameters_t& param)
    : _param(param),
    _ready(false)
{
    if (N != param.selected_columns.size())
        throw std::logic_error("Inconsistent number of features.");
}


template<std::size_t N>
bool Dataset<N>::check_index(const int index)
{
	return std::find(
			_param.selected_columns.begin(),
			_param.selected_columns.end(),
			index
		) != _param.selected_columns.end();
}


template<std::size_t N>
bool Dataset<N>::check_file()
{
	if (!std::filesystem::exists(_param.filename))
	{
		std::cerr << "File doesn't exists." << std::endl;
		return false;
	}
	return true;
}


/*** RowMajorDataset ***/

template<std::size_t N>
RowMajorDataset<N>::RowMajorDataset(const parser_parameters_t& param)
	: Dataset<N>(param)
{ }


template<std::size_t N>
std::vector<std::array<std::string, N>>
RowMajorDataset<N>::get_samples()
{
	if (this->_ready)
        return _samples;
    throw std::logic_error("Dataset must be imported before using samples.");
}


template<std::size_t N>
bool RowMajorDataset<N>::import()
{
	int row_counter = 0;
    const std::size_t sample_size = this->_param.selected_columns.size();
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
		int column = 0, sample_column = 0;
		std::array<std::string, N> sample;
		
		pos_end = current_sample.find(separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (this->check_index(column))
			{
				sample[sample_column++] =
                    current_sample.substr(pos_start, pos_end - pos_start);
			}
			pos_start = pos_end + 1;
			pos_end = current_sample.find(separator, pos_start);
			column++;
		}
		if (this->check_index(column))
		{
			sample[sample_column] = std::move(
                current_sample.substr(pos_start, pos_end - pos_start)
            );
		}
		_samples.push_back(std::move(sample));
		row_counter++;
	}
    this->_ready = true;
	return true;
}


template<std::size_t N>
void RowMajorDataset<N>::print()
{
	int i = 1;
	for (const auto& sample : this->_samples)
	{
		std::printf("%i\t: ", i++);
		for (const auto& feature : sample)
			std::cout << feature << "\t";
		std::cout << std::endl;
	}
}


/*** ColMajorDataset ***/

template<std::size_t N>
ColMajorDataset<N>::ColMajorDataset(const parser_parameters_t& param)
	: Dataset<N>(param)
{ }


template<std::size_t N>
std::array<std::vector<std::string>, N>
ColMajorDataset<N>::get_features()
{
	if (this->_ready)
        return _features;
    throw std::logic_error("Dataset must be imported before using samples.");
}


template<std::size_t N>
bool ColMajorDataset<N>::import()
{
	int row_counter = 0;
    const std::size_t sample_size = this->_param.selected_columns.size();
	const int max = this->_param.max_samples;
	const char separator = this->_param.separator;
	std::fstream file(this->_param.filename);
	std::string current_sample;

	if (!this->check_file())
		return false;

	for (auto& feature : _features)
		feature.clear();

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


template<std::size_t N>
void ColMajorDataset<N>::print()
{
	int i = 1;
	const int max = _features[0].size();
	for (int i = 0; i < max; i++)
	{
		std::printf("%i\t: ", i);
		for (int j = 0; j<N; j++)
			std::cout << _features[j][i] << "\t";
		std::cout << std::endl;
	}
}

#endif // #ifndef __DATASET_HPP__