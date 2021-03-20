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

template<std::size_t N_FEATURES>
class Dataset
{
private:
	parser_parameters_t _param;
	std::vector<std::array<std::string, N_FEATURES>> _samples;
	bool _is_ready;

public:
	Dataset(const parser_parameters_t& param);
	Dataset() {}
	void import();
	void print();
	std::vector<std::array<std::string, N_FEATURES>> get_samples();
};


/*** DEFINITIONS ***/

template<std::size_t N_FEATURES>
Dataset<N_FEATURES>::Dataset(const parser_parameters_t& param)
    : _param(param),
    _is_ready(false)
{
    if (N_FEATURES != param.selected_columns.size())
        throw std::logic_error("Inconsistent number of features.");
}

template<std::size_t N_FEATURES>
void Dataset<N_FEATURES>::import()
{
	int row_counter = 0;
    const std::size_t sample_size = _param.selected_columns.size();
	std::fstream file(_param.filename);
	std::string current_sample;
    _samples.clear();
	const auto& selected_columns = _param.selected_columns;

	if (!std::filesystem::exists(_param.filename))
	{
		std::cerr << "File doesn't exists." << std::endl;
		return;
	}

	auto check_index = [&selected_columns](const int index) {
		return std::find(
				selected_columns.begin(),
				selected_columns.end(),
				index) != selected_columns.end();
	};
	while (row_counter < _param.max_samples && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column = 0, sample_column = 0;
		std::array<std::string, N_FEATURES> sample;
		
		pos_end = current_sample.find(_param.separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (check_index(column))
				sample[sample_column++] =
                    current_sample.substr(pos_start, pos_end - pos_start);
			pos_start = pos_end + 1;
			pos_end = current_sample.find(_param.separator, pos_start);
			column++;
		}
		if (check_index(column))
			sample[sample_column] = std::move(
                current_sample.substr(pos_start, pos_end - pos_start)
            );
		_samples.push_back(std::move(sample));
		row_counter++;
	}
    _is_ready = true;
}

template<std::size_t N_FEATURES>
std::vector<std::array<std::string, N_FEATURES>>
Dataset<N_FEATURES>::get_samples()
{
    if (_is_ready)
        return _samples;
    throw std::logic_error("Dataset must be imported before using samples.");
}

template<std::size_t N_FEATURES>
void Dataset<N_FEATURES>::print()
{
	int i = 1;
	for (const auto& sample : _samples)
	{
		std::printf("%i\t: ", i++);
		for (const auto& feature : sample)
			std::cout << feature << '\t';
		std::cout << std::endl;
	}
}