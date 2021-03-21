#ifndef __CHRONOMETER_HPP__
#define __CHRONOMETER_HPP__

#include <chrono>
#include <vector>
#include <numeric>

class Chronometer
{
private:
    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _last;
    std::vector<double> _laps;

public:
    Chronometer()
    {
        start();
    }

    void start()
    {
        _start = std::chrono::steady_clock::now();
        _last = _start;
        _laps.clear();
    }

    double lap()
    {
        auto tick = std::chrono::steady_clock::now();
        std::chrono::duration<double> lap = tick - _last;
        _laps.push_back(lap.count());
        _last = tick;
        return lap.count();
    }

    double elapsed()
    {
        auto tick = std::chrono::steady_clock::now();
        std::chrono::duration<double> elap = tick - _start;
        return elap.count();
    }

    double average()
    {
        return std::accumulate(_laps.begin(), _laps.end(), 0.0)
            / static_cast<double>(_laps.size());
    }
};

#endif // #ifndef __CHRONOMETER_HPP__