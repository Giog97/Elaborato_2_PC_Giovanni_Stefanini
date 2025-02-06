#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA
__global__ void helloFromGPU()
{
    printf("Hello from GPU!\n");
}

// Funzione chiamabile da main.cpp
void runCUDA()
{
    std::cout << "Hello from CPU (before GPU)!" << std::endl;

    // Lancia il kernel CUDA
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize(); // Aspetta la fine dell'esecuzione della GPU

    std::cout << "Hello from CPU (after GPU)!" << std::endl;
}

