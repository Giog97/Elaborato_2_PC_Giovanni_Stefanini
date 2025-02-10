//
// Created by giost on 05/02/2025.
//

#include "histogram_equalization_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

// Kernel 1 per calcolare l'istogramma con memoria condivisa
__global__ void computeHistogram(const uchar* input, int* hist, int width, int height) {
    __shared__ int local_hist[256]; // Memoria condivisa:si usa memoria shared per accumulare un istogramma locale (migliora le performance, riducendo il traffico con la memoria globale)
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // Serve per indicizzare l'array dell'istogramma.

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
        int pixel_value = input[y * width + x]; // Accesso coalescente
        atomicAdd(&local_hist[pixel_value], 1);
    }
    __syncthreads();

    // Unisci gli istogrammi locali in quello globale
    if (tid < 256) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

// Kernel 2 per calcolare la CDF
// Si usa un Parallel Prefix Sum (Scan) per velocizzare il calcolo della CDF
__global__ void computeCDF(int* hist, int* cdf) {
    __shared__ int temp[256]; // Memoria condivisa

    int tid = threadIdx.x;

    // Carica l'istogramma nella memoria condivisa
    temp[tid] = hist[tid];
    __syncthreads();

    // **Up-sweep (Riduzione)**
    for (int offset = 1; offset < 256; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Scrivi il risultato finale
    cdf[tid] = temp[tid];
}

// Kernel 3 per applicare la trasformazione
__global__ void applyTransformation(uchar* output, const uchar* input, const uchar* lookup_table, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = lookup_table[input[idx]]; // Accesso coalescente
    }
}

void histogram_equalization_cuda(const Mat& input, Mat& output) {
    int width = input.cols; // Larghezza sarà quella della immagine che viene passata in input (colonne delle img)
    int height = input.rows; // Altezza sarà quella della immagine che viene passata in input (righe delle img)
    int total_pixels = width * height;

    // Alloca memoria sulla GPU (device) (NB: h = host, d = device)
    uchar* d_input;
    uchar* d_output;
    int* d_hist; // Istogramma
    int* d_cdf; // CDF
    uchar* d_lookup_table; // Tabella di lookup
    // In questo modo si ha che dati di ogni attributo sono separati in array distinti + efficace per parallelismo su GPU xché permette accesso coalescente a memoria
    // --> Ogni struttura dati è rappresentata come un array indipendente.

    cudaMalloc((void**)&d_input, total_pixels * sizeof(uchar));
    cudaMalloc((void**)&d_output, total_pixels * sizeof(uchar));
    cudaMalloc((void**)&d_hist, 256 * sizeof(int));
    cudaMalloc((void**)&d_cdf, 256 * sizeof(int));
    cudaMalloc((void**)&d_lookup_table, 256 * sizeof(uchar));

    // Usa Pinned Memory per le strutture sulla CPU (host) (NB: h = host, d = device) --> in modo da velocizzare i trasferimenti da GPU a CPU
    int* h_cdf;
    uchar* h_lookup_table;
    cudaHostAlloc((void**)&h_cdf, 256 * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_lookup_table, 256 * sizeof(uchar), cudaHostAllocDefault);

    // Copia l'immagine input sulla GPU
    cudaMemcpy(d_input, input.data, total_pixels * sizeof(uchar), cudaMemcpyHostToDevice);

    // Inizializza l'istogramma a zero
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Definizione eventi CUDA per il timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Parametri per i kernel
    dim3 blockSize(32, 16); // Dimensione del blocco è data da 32x16 (512 threads) // Provando a cambiare con 32x32 non cambia molto
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Inizia la misurazione del tempo solo per i kernel
    cudaEventRecord(start);

    // Kernel 1: Calcolo dell'istogramma
    computeHistogram<<<gridSize, blockSize>>>(d_input, d_hist, width, height);
    //cudaDeviceSynchronize(); // Sincronizza prima di passare alla CDF --> rimossa perché non necessaria

    // Kernel 2: Calcolo della CDF
    computeCDF<<<1, 256>>>(d_hist, d_cdf);
    //cudaDeviceSynchronize(); // Sincronizza prima di passare alla applyTransformation --> rimossa perché non necessaria

    // Copia la CDF dalla GPU alla CPU (più veloce grazie alla Pinned Memory) [senza sincronizzazione]
    cudaMemcpyAsync(h_cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Aspetta che la CDF sia copiata sulla CPU prima di procedere con la lookup table
    cudaDeviceSynchronize(); // Questa sincronizzazione è necessaria per evitare che la lookup table venga calcolata prima che h_cdf sia pronto

    // Calcolo della lookup table sulla CPU
    int min_cdf = h_cdf[0];
    for (int i = 1; i < 256; i++) {
        if (h_cdf[i] < min_cdf) {
            min_cdf = h_cdf[i];
        }
    }

    for (int i = 0; i < 256; i++) {
        float value = ((h_cdf[i] - min_cdf) * 255.0f) / (total_pixels - min_cdf);
        h_lookup_table[i] = static_cast<uchar>(std::min(std::max(value, 0.0f), 255.0f));
    }

    // Copia la lookup table sulla GPU (più veloce grazie alla Pinned Memory)
    cudaMemcpyAsync(d_lookup_table, h_lookup_table, 256 * sizeof(uchar), cudaMemcpyHostToDevice);

    // Kernel 3: Applicazione della trasformazione (applica la tabella di lookup all'immagine)
    applyTransformation<<<gridSize, blockSize>>>(d_output, d_input, d_lookup_table, width, height);

    // Sincronizza solo prima di operazioni critiche, non per ogni kernel
    cudaDeviceSynchronize();

    // Registra il tempo di fine
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcola il tempo impiegato dai kernel
    float kernel_time;
    // Si prende il tempo passato tra i due eventi start e stop
    cudaEventElapsedTime(&kernel_time, start, stop); // Serve calcolare tempo totale esecuzione dei 3 kernel senza considerare copie di memoria (cudaMemcpy) --> prende il tmepo tra i due eventi (start e stop)
    std::cout << "Tempo di esecuzione solo dei kernel CUDA: " << kernel_time << " ms" << std::endl;

    // Copia il risultato sulla CPU (più veloce grazie alla Pinned Memory)
    cudaMemcpy(output.data, d_output, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost); // Questa operazione è quella che rallenta l'esecuzione

    // Libera memoria GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaFree(d_lookup_table);

    // Libera la memoria pinned sulla CPU
    cudaFreeHost(h_cdf);
    cudaFreeHost(h_lookup_table);

    // Distrugge gli eventi CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



