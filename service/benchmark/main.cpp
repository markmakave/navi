#include "lumina.hpp"

#include <fstream>
#include <cstring>

void memory();

int main() {

    memory(); 

    return 0;
};

#define ITERATIONS 100

void
memory() {

    std::ofstream bench_file("memory_bench.csv");
    bench_file << "Bytes,Bandwidth\n";

    constexpr size_t min = 1024;
    constexpr size_t max = 1024*1024*1024;

    void *src_ptr, *dst_ptr;

    // H2D
    {
        src_ptr = new uint8_t[max];
        dst_ptr = lumina::cuda::malloc(max);

        std::memset(src_ptr, 0, max);

        for (size_t bytes = min; bytes <= max; bytes *= 2) {
            lumina::timer t("memcopy");
            for (int i = 0; i < ITERATIONS; ++i) {
                lumina::cuda::memcpy(dst_ptr, src_ptr, bytes, lumina::cuda::H2D);
            }
            auto elapsed = t.elapsed() / ITERATIONS;

            double bandwidth = (bytes / 1024.0 / 1024.0) / elapsed;

            bench_file << bytes << ',' << bandwidth << '\n';
            lumina::log::info("Passed", bytes, "bytes bench");
        }

        delete[] src_ptr;
        lumina::cuda::free(dst_ptr);
    }

    // D2H
    {
        src_ptr = lumina::cuda::malloc(max);
        dst_ptr = new uint8_t[max];
    
        std::memset(dst_ptr, 0, max);
    
        for (size_t bytes = min; bytes <= max; bytes *= 2) {
            lumina::timer t("memcopy");
            for (int i = 0; i < ITERATIONS; ++i) {
                lumina::cuda::memcpy(dst_ptr, src_ptr, bytes, lumina::cuda::D2H);
            }
            auto elapsed = t.elapsed() / ITERATIONS;

            double bandwidth = (bytes / 1024.0 / 1024.0) / elapsed;

            bench_file << bytes << ',' << bandwidth << '\n';
            lumina::log::info("Passed", bytes, "bytes bench");
        }

        lumina::cuda::free(src_ptr);
        delete[] dst_ptr; 
    }

    // D2D
    {
        src_ptr = lumina::cuda::malloc(max);
        dst_ptr = lumina::cuda::malloc(max);

        for (size_t bytes = min; bytes <= max; bytes *= 2) {
            lumina::timer t("memcopy");
            for (int i = 0; i < ITERATIONS; ++i) {
                lumina::cuda::memcpy(dst_ptr, src_ptr, bytes, lumina::cuda::D2D);
            }
            auto elapsed = t.elapsed() / ITERATIONS;

            double bandwidth = (bytes / 1024.0 / 1024.0) / elapsed;

            bench_file << bytes << ',' << bandwidth << '\n';
            lumina::log::info("Passed", bytes, "bytes bench");
        }

        lumina::cuda::free(src_ptr);
        lumina::cuda::free(dst_ptr);
    }

    // H2H
    {
        src_ptr = new uint8_t[max];
        dst_ptr = new uint8_t[max];

        std::memset(src_ptr, 0, max);
        std::memset(dst_ptr, 0, max);

        for (size_t bytes = min; bytes <= max; bytes *= 2) {
            lumina::timer t("memcopy");
            for (int i = 0; i < ITERATIONS; ++i) {
                lumina::cuda::memcpy(dst_ptr, src_ptr, bytes, lumina::cuda::H2H);
            }
            auto elapsed = t.elapsed() / ITERATIONS;

            double bandwidth = (bytes / 1024.0 / 1024.0) / elapsed;

            bench_file << bytes << ',' << bandwidth << '\n';
            lumina::log::info("Passed", bytes, "bytes bench");
        }

        delete[] src_ptr; 
        delete[] dst_ptr; 
    }

    // MCPY
    {
        src_ptr = new uint8_t[max];
        dst_ptr = new uint8_t[max];

        std::memset(src_ptr, 0, max);
        std::memset(dst_ptr, 0, max);

        for (size_t bytes = min; bytes <= max; bytes *= 2) {
            lumina::timer t("memcopy");
            for (int i = 0; i < ITERATIONS; ++i) {
                std::memcpy(dst_ptr, src_ptr, bytes);
            }
            auto elapsed = t.elapsed() / ITERATIONS;

            double bandwidth = (bytes / 1024.0 / 1024.0) / elapsed;

            bench_file << bytes << ',' << bandwidth << '\n';
            lumina::log::info("Passed", bytes, "bytes bench");
        }

        delete[] src_ptr; 
        delete[] dst_ptr; 
    }
}
