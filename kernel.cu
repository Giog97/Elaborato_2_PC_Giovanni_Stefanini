//
// Created by giost on 05/02/2025.
//

#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

// Kernel 1 per calcolare l'istogramma con memoria condivisa
__global__ void computeHistogram(const uchar* input, int* hist, int width, int height) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Inizializza l'istogramma locale
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Calcola le coordinate globali
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Aggiorna l'istogramma locale
    if (x < width && y < height) {
        int pixel_value = input[y * width + x];
        atomicAdd(&local_hist[pixel_value], 1);
    }
    __syncthreads();

    // Unisci gli istogrammi locali in quello globale
    if (tid < 256) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

// Kernel 2 per calcolare la CDF
__global__ void computeCDF(int* hist, int* cdf) {
    int idx = threadIdx.x;
    int temp = 0;
    for (int i = 0; i <= idx; i++) {
        temp += hist[i];
    }
    cdf[idx] = temp;
}

// Kernel 3 per applicare la trasformazione
__global__ void applyTransformation(uchar* output, const uchar* input, const uchar* lookup_table, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = lookup_table[input[idx]];
    }
}

void histogram_equalization_cuda(const Mat& input, Mat& output) {
    int width = input.cols;
    int height = input.rows;
    int total_pixels = width * height;

    // Alloca memoria sulla GPU
    uchar* d_input;
    uchar* d_output;
    int* d_hist;
    int* d_cdf;
    uchar* d_lookup_table;

    cudaMalloc((void**)&d_input, total_pixels * sizeof(uchar));
    cudaMalloc((void**)&d_output, total_pixels * sizeof(uchar));
    cudaMalloc((void**)&d_hist, 256 * sizeof(int));
    cudaMalloc((void**)&d_cdf, 256 * sizeof(int));
    cudaMalloc((void**)&d_lookup_table, 256 * sizeof(uchar));

    // Copia l'immagine input sulla GPU
    cudaMemcpy(d_input, input.data, total_pixels * sizeof(uchar), cudaMemcpyHostToDevice);

    // Inizializza l'istogramma a zero
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Definizione eventi CUDA per il timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Parametri per i kernel
    dim3 blockSize(32, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Inizia la misurazione del tempo solo per i kernel
    cudaEventRecord(start);

    // Kernel 1: Calcolo dell'istogramma
    computeHistogram<<<gridSize, blockSize>>>(d_input, d_hist, width, height);
    cudaDeviceSynchronize(); // Sincronizza prima di passare alla CDF

    // Kernel 2: Calcolo della CDF
    computeCDF<<<1, 256>>>(d_hist, d_cdf);
    cudaDeviceSynchronize(); // Sincronizza prima di passare alla applyTransformation


    // Copia la CDF dalla GPU alla CPU
    int h_cdf[256];
    cudaMemcpy(h_cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Calcolo della lookup table sulla CPU
    int min_cdf = h_cdf[0];
    for (int i = 1; i < 256; i++) {
        if (h_cdf[i] < min_cdf) {
            min_cdf = h_cdf[i];
        }
    }

    uchar h_lookup_table[256];
    for (int i = 0; i < 256; i++) {
        float value = ((h_cdf[i] - min_cdf) * 255.0f) / (total_pixels - min_cdf);
        h_lookup_table[i] = static_cast<uchar>(std::min(std::max(value, 0.0f), 255.0f));
    }

    // Copia la lookup table sulla GPU
    cudaMemcpy(d_lookup_table, h_lookup_table, 256 * sizeof(uchar), cudaMemcpyHostToDevice);

    // Kernel 3: Applicazione della trasformazione
    applyTransformation<<<gridSize, blockSize>>>(d_output, d_input, d_lookup_table, width, height);
    cudaDeviceSynchronize();

    // Registra il tempo di fine
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato dai kernel
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop); // Serve calcolare tempo totale esecuzione dei 3 kernel senza considerare copie di memoria (cudaMemcpy)
    std::cout << "Tempo di esecuzione solo dei kernel CUDA: " << kernel_time << " ms" << std::endl;

    // Copia il risultato sulla CPU
    cudaMemcpy(output.data, d_output, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Libera memoria GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaFree(d_lookup_table);

    // Distrugge gli eventi CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



