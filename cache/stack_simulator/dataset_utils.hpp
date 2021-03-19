#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

using std::endl;

using dataset_t = std::vector<std::vector<std::string>>;

typedef struct parser_parameters
{
	std::vector<int> selected_columns;
	int max_samples;
	char separator;
	std::string filename;
} parser_parameters_t;

/*** DECLARATIONS ***/

dataset_t import_dataset(const parser_parameters_t& param);
void print_dataset(dataset_t dataset);

/*** DEFINITIONS ***/

dataset_t import_dataset(const parser_parameters_t& param)
{
	int count = 0;
	std::fstream file(param.filename);
	std::string current_sample;
	dataset_t dataset;
	const auto& selected_columns = param.selected_columns;

	auto check_index = [&selected_columns](int index) {
		return std::find(
				selected_columns.begin(),
				selected_columns.end(),
				index) != selected_columns.end();
	};
	while (count < param.max_samples && std::getline(file, current_sample))
	{
		int pos_start = 0, pos_end = 0;
		int column_index = 0;
		std::vector<std::string> sample;
		pos_end = current_sample.find(param.separator, pos_start);
		
		while (pos_end != std::string::npos)
		{
			if (check_index(column_index))
				sample.push_back(current_sample.substr(pos_start, pos_end - pos_start));
			pos_start = pos_end + 1;
			pos_end = current_sample.find(param.separator, pos_start);
			column_index++;
		}
		if (check_index(column_index))
			sample.push_back(current_sample.substr(pos_start, pos_end - pos_start));
		dataset.push_back(sample);
		count++;
	}

	return dataset;
}

void print_dataset(dataset_t dataset)
{
	int i = 1;
	for (auto& sample : dataset)
	{
		std::printf("%i\t: ", i++);
		for (auto& feature : sample)
			std::cout << feature << '\t';
		std::cout << endl;
	}
}