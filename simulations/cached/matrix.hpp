#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <vector>
#include <iostream>

// Implementation of a simple row-major 2d vector
// The purpose is to avoid the double dereferentiation of the common
// vector of vectors

template<typename T>
class Matrix
{
public:
    Matrix(uint32_t N, uint32_t M)
        : _matrix(std::vector<T>(N*M))
    {
        this->N = N;
        this->M = M;
    }

    T& at(uint32_t i, uint32_t j)
    {
        return _matrix[i*N + j];
    }

    void print(std::ostream& stream, char sep = '\t', char newline = '\n')
    {
        for (uint32_t i = 0; i < N; ++i)
        {
            for (uint32_t j = 0; j < M - 1; ++j)
            {
                stream << _matrix[i*N + j] << sep;
            }
            stream << _matrix[i*N + M - 1] << newline;
        }
    }

    void print(char sep = '\t', char newline = '\n')
    {
        this->print(std::cout, sep, newline);
    }

    uint32_t N, M;

private:
    std::vector<T> _matrix;
};

#endif // #ifndef __MATRIX_HPP__