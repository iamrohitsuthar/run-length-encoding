#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>
using in_elt_t = int;

template<typename elt_t>
struct array
{
	elt_t *data;
	size_t size; // the number of elt_t elements in data

	static array<elt_t> new_on_device(size_t size)
	{
		array<elt_t> d_result{nullptr, size};
		d_result.cudaMalloc();
		return d_result;
	}

	static array<elt_t> vector_view_on_host(std::vector<elt_t> &v)
	{
		return array<elt_t>{v.data(), v.size()};
	}

	array<elt_t> subview(size_t offset, size_t subview_size)
	{
		size_t result_size = std::min(subview_size, size - offset);
		return array<elt_t>{data + offset, result_size};
	}

	elt_t &operator[](const size_t i)
	{
		return data[i];
	}

	void cudaMalloc()
	{
		checkCuda(::cudaMalloc(&data, size * sizeof(*data)));
	}

	void cudaFree()
	{
		checkCuda(::cudaFree(data));
	}
};

void append_partial_result(std::vector<in_elt_t> &out_symbols, std::vector<int> &out_counts, std::vector<in_elt_t> &full_out_symbols, std::vector<int> &full_out_counts) {
	size_t offset = 0;

	if (full_out_symbols.size() > 0 && out_symbols.size() > 0) {
		size_t prev_full_end{out_symbols.size() - 1};
		if (full_out_symbols[prev_full_end] == out_symbols[0]) {
			full_out_counts[prev_full_end] += out_counts[0];
			offset = 1;
		}
	}

	std::copy(out_symbols.begin() + offset, out_symbols.end(), std::back_inserter(full_out_symbols));
	std::copy(out_counts.begin() + offset, out_counts.end(), std::back_inserter(full_out_counts));
}

int serial_rle_helper(const in_elt_t* in, int n, in_elt_t* symbolsOut, int* countsOut) {
	if (n == 0)
		return 0; // nothing to compress!

	int outIndex = 0;
	in_elt_t symbol = in[0];
	int count = 1;

	for (int i = 1; i < n; ++i) {
		if (in[i] != symbol) {
			// run is over.
			// So output run.
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			// and start new run:
			symbol = in[i];
			count = 1;
		} else {
			++count; // run is not over yet.
		}
	}

	// output last run.
	symbolsOut[outIndex] = symbol;
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}

void serial_rle(array<in_elt_t> in, std::vector<in_elt_t> &out_symbols, std::vector<int> &out_counts, int &out_end) {
	out_end = serial_rle_helper(in.data, in.size, out_symbols.data(), out_counts.data());
}

void run_rle_impl(array<in_elt_t> in, std::vector<in_elt_t> &out_symbols, std::vector<int> &out_counts, int &out_end, bool use_cpu_impl) {
	if (use_cpu_impl)
		serial_rle(in, out_symbols, out_counts, out_end);
}

void rle(std::vector<in_elt_t> &in_owner, std::vector<in_elt_t> &full_out_symbols, std::vector<int> &full_out_counts, size_t piece_size, bool use_cpu_impl) {
	array<in_elt_t> full_in = array<in_elt_t>::vector_view_on_host(in_owner);

	for (size_t start = 0; start < in_owner.size(); start += piece_size) {
		array<in_elt_t> in = full_in.subview(start, piece_size);
		std::cout << "Partial in start: " << start
				  << ", size: " << in.size << std::endl;

		// TODO Could actually be allocated once
		std::vector<in_elt_t> out_symbols(in.size);
		std::vector<int> out_counts(in.size);
		int end{0};

		run_rle_impl(in, out_symbols, out_counts, end, use_cpu_impl);

		out_symbols.resize(end);
		out_counts.resize(end);

		append_partial_result(out_symbols, out_counts, full_out_symbols, full_out_counts);
	}
}

int main(int argc, char *argv[]) {
    bool use_cpu_impl = true;
    size_t input_size = 8;

    if (use_cpu_impl)
		std::cout<<"Using the CPU implementation"<<std::endl;
    else
		std::cout<<"Using the GPU implementation"<<std::endl;

    std::cout<<"Creating Input..."<<std::endl;
    std::vector<in_elt_t> input{};
    input.push_back(1);
    input.push_back(2);
    input.push_back(3);
    input.push_back(6);
    input.push_back(6);
    input.push_back(6);
    input.push_back(5);
    input.push_back(5);

    std::cout<<"Initial Input: "<<std::endl;
    std::cout<<"[";
    for(int i = 0 ; i < input.size() ; i++) {
        std::cout<<input[i]<<" ";
    }
    std::cout<<"]";
    std::cout<<std::endl;

    std::vector<in_elt_t> out_symbols{};
	std::vector<int> out_counts{};
    
    rle(in_owner, out_symbols, out_counts, input_piece_size, use_cpu_impl);

    std::cout<<"Output Symbols: "<<std::endl;
    std::cout<<"[";
    for(int i = 0 ; i < out_symbols.size() ; i++) {
        std::cout<<out_symbols[i]<<" ";
    }
    std::cout<<"]";
    std::cout<<std::endl;
    std::cout<<"Count: "<<std::endl;
    std::cout<<"[";
    for(int i = 0 ; i < out_counts.size() ; i++) {
        std::cout<<out_counts[i]<<" ";
    }
    std::cout<<"]";
    std::cout<<std::endl;

    return 0;
}