//
// Created by giost on 05/02/2025.
//

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

// Kernel per calcolare l'istogramma utilizzando la memoria condivisa
__global__ void computeHistogram(const uchar* input, int* hist, int width, int height) {
    // Ogni blocco calcola un istogramma locale nella memoria condivisa
    __shared__ int local_hist[256];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

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

// Kernel per calcolare la CDF
__global__ void computeCDF(int* hist, int* cdf) {
    int idx = threadIdx.x;
    int temp = 0;
    for (int i = 0; i <= idx; i++) {
        temp += hist[i];
    }
    cdf[idx] = temp;
}

// Kernel per applicare la trasformazione utilizzando la lookup table
__global__ void applyTransformation(uchar* output, const uchar* input, const uchar* lookup_table, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = lookup_table[input[idx]];
    }
}

// Funzione principale per l'equalizzazione dell'istogramma con CUDA
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

    // Parametri per i kernel
    dim3 blockSize(32, 8); // 256 thread per blocco (32x8)
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Calcola l'istogramma
    computeHistogram<<<gridSize, blockSize>>>(d_input, d_hist, width, height);

    // Calcola la CDF
    computeCDF<<<1, 256>>>(d_hist, d_cdf);

    // Copia la CDF dalla GPU alla CPU
    int h_cdf[256];
    cudaMemcpy(h_cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Calcola min_cdf dalla CDF copiata
    int min_cdf = h_cdf[0];
    for (int i = 1; i < 256; i++) {
        if (h_cdf[i] < min_cdf) {
            min_cdf = h_cdf[i];
        }
    }

    // Calcola la lookup table
    uchar h_lookup_table[256];
    if (total_pixels == min_cdf) {
        // Evita divisione per zero (caso teorico)
        memset(h_lookup_table, 0, 256 * sizeof(uchar));
    } else {
        for (int i = 0; i < 256; i++) {
            float value = ((h_cdf[i] - min_cdf) * 255.0f) / (total_pixels - min_cdf);
            h_lookup_table[i] = static_cast<uchar>(std::min(std::max(value, 0.0f), 255.0f));
        }
    }

    // Copia la lookup table sulla GPU
    cudaMemcpy(d_lookup_table, h_lookup_table, 256 * sizeof(uchar), cudaMemcpyHostToDevice);

    // Applica la trasformazione
    applyTransformation<<<gridSize, blockSize>>>(d_output, d_input, d_lookup_table, width, height);

    // Copia il risultato sulla CPU
    cudaMemcpy(output.data, d_output, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Libera la memoria della GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaFree(d_lookup_table);
}

