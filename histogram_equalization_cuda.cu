//
// Created by gioste.
//

#include "histogram_equalization_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

// Kernel 1 per calcolare l'istogramma con memoria condivisa
__global__ void computeHistogram(const uchar* input, int* hist, int width, int height) {
    __shared__ int local_hist[256];  // Istogramma locale in memoria shared, per accumulare un istogramma locale (migliora le performance, riducendo il traffico con la memoria globale)
    __shared__ uchar tile[16][16];   // Memoria shared per un tile 16x16
    // Ogni blocco di thread gestisce una porzione dell'immagine (un "tile" di 16x16 pixel)

    // Calcolo indice lineare del thread (identificatore unico per ogni thread all'interno di un blocco)
    // threadIdx.x → Indice orizzontale del thread all'interno del blocco.
    // threadIdx.y * blockDim.x → Offset verticale del thread, così thread delle righe successive hanno indici consecutivi.
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // Indice lineare del thread --> Serve per indicizzare l'array dell'istogramma.

    // Inizializza l'istogramma locale
    if (tid < 256) {
        local_hist[tid] = 0; // Valori dell'istogramma locale vengono inizializzati a zero
    }
    __syncthreads();

    // Calcolo coordinate globali (ie posizione assoluta rispetto all'intera immagine)
    // blockIdx.x * blockDim.x → Offset del blocco rispetto all'origine dell'immagine.
    // threadIdx.x → Offset del thread all'interno del blocco.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Stessa cosa di x

    // Carichiamo i pixel nella shared memory per il Tile (Utilizzo della memoria shared per i pixel dell'immagine)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x]; // Ogni thread carica un pixel dalla memoria globale alla memoria condivisa
    }
    __syncthreads(); // All threads in the same block must reach the __syncthreads() before any of the them can move on

    // Calcolo dell'istogramma locale direttamente sulla shared memory
    if (x < width && y < height) {
        int pixel_value = tile[threadIdx.y][threadIdx.x]; // Ogni thread legge un valore di intensità dal tile e poi ...
        atomicAdd(&local_hist[pixel_value], 1); // ... Ogni thread aggiorna l'istogramma locale usando un'operazione atomica
    }
    __syncthreads(); // All threads in the same block must reach the __syncthreads() before any of the them can move on

    // Uniamo i risultati dell'istogramma locale con la memoria globale (Scrittura ottimizzata della memoria globale)
    if (tid < 256) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

// Kernel 2 per calcolare la CDF
// Si usa un Parallel Reduce per velocizzare il calcolo della CDF
__global__ void computeCDF(int* hist, int* cdf) {
    __shared__ int temp[256]; // Memoria condivisa

    int tid = threadIdx.x; // Indice del thread all'interno del blocco, che va da 0 a 255. Indicizza memoria condivisa e determina quali thread eseguono le somme durante la riduzione parallela

    // Carica l'istogramma nella memoria condivisa
    temp[tid] = hist[tid]; // Valori istogramma caricati in un array condiviso (temp[256])
    __syncthreads(); // Assicura che tutti i valori siano caricati prima di procedere oltre

    // Riduzione Parallela
    for (int offset = 1; offset < 256; offset *= 2) {
        // Offset raddoppia perchè  così ho somma cumulativa in un numero logaritmico di passi
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset]; // Legge il valore precedente a distanza offset e lo somma al proprio valore
        } // Se tid < offset, il thread non modifica il proprio valore in questo step
        __syncthreads(); // Per evitare letture/scritture incoerenti
        temp[tid] += val; // Somma risultante viene salvata come CDF
        __syncthreads(); // Per evitare letture/scritture incoerenti
    }

    // Scrivi il risultato finale
    cdf[tid] = temp[tid]; // CDF calcolata viene scritta in memoria globale
}

// Kernel 3 per applicare la trasformazione
__global__ void applyTransformation(uchar* output, const uchar* input, const uchar* lookup_table, int width, int height) {
    __shared__ uchar tile[16][16]; // Memoria condivisa per un Tile 16x16 // Se qui metto 8x8 l'immagine si sciupa

    // Calcolo delle coordinate globali
    // blockIdx.x * blockDim.x → Offset del blocco rispetto all'origine dell'immagine.
    // threadIdx.x → Offset del thread all'interno del blocco.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Carichiamo un Tile nella shared memory (copia dalla mem globale alla mem condivisa)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x]; // Dati immagine caricati in memoria shared per migliorare accesso
    }
    __syncthreads(); // All threads in the same block must reach the __syncthreads() before any of the them can move on

    // Applicazione della trasformazione
    if (x < width && y < height) {
        // Ogni thread legge valore di un pixel dal tile shered e lo sostituisce con valore corrispondente nella lookup table
        output[y * width + x] = lookup_table[tile[threadIdx.y][threadIdx.x]]; // Accesso Coalescente
    }
}

void histogram_equalization_cuda(const Mat& input, Mat& output) {
    int width = input.cols; // Larghezza sarà quella della immagine che viene passata in input (colonne delle img)
    int height = input.rows; // Altezza sarà quella della immagine che viene passata in input (righe delle img)
    int total_pixels = width * height;

    // Informazioni utili per il calcolo dell'Occupancy (Kernel 1 computeHistogram)
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)computeHistogram);
    std::cout << "Registri per thread (kernel 1): " << attr.numRegs << std::endl;
    std::cout << "Shared memory per blocco (kernel 1): " << attr.sharedSizeBytes << " bytes" << std::endl;

    // Informazioni utili per il calcolo dell'Occupancy (Kernel 2 computeCDF)
    cudaFuncAttributes attr2;
    cudaFuncGetAttributes(&attr2, (const void*)computeCDF);
    std::cout << "Registri per thread (kernel 2): " << attr2.numRegs << std::endl;
    std::cout << "Shared memory per blocco (kernel 2): " << attr2.sharedSizeBytes << " bytes" << std::endl;

    // Informazioni utili per il calcolo dell'Occupancy (Kernel 3 applyTransformation)
    cudaFuncAttributes attr3;
    cudaFuncGetAttributes(&attr3, (const void*)applyTransformation);
    std::cout << "Registri per thread (kernel 3): " << attr3.numRegs << std::endl;
    std::cout << "Shared memory per blocco (kernel 3): " << attr3.sharedSizeBytes << " bytes" << std::endl;

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
    cudaMemcpy(d_input, input.data, total_pixels * sizeof(uchar), cudaMemcpyHostToDevice); // Operazione che rallenta l'esecuzione

    // Inizializza l'istogramma a zero
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Definizione eventi CUDA per il timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Parametri per i kernel
    dim3 blockSize(16, 16); // Dimensione del blocco è data da 16x16 (256 threads) // Provato a cambiare con 32x32 non cambia molto // Poi con il calcolo dell'occupancy è una dimensione buona
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y); // GridSize dipende dall'immagine

    // Inizia la misurazione del tempo solo per i kernel
    cudaEventRecord(start);

    // Kernel 1: Calcolo dell'istogramma
    computeHistogram<<<gridSize, blockSize>>>(d_input, d_hist, width, height);
    //cudaDeviceSynchronize(); // Sincronizza prima di passare alla CDF --> rimossa perché non necessaria (rallenta esecuzione)
    // CUDA garantisce che il kernel 2 leggerà d_hist (memoria globale GPU) solo dopo che il kernel 1 ha scritto tutti i dati.

    // Kernel 2: Calcolo della CDF
    computeCDF<<<1, 256>>>(d_hist, d_cdf); // In questo ho 256 thread che lavoreranno insieme all'interno dello stesso blocco
    //cudaDeviceSynchronize(); // Sincronizza prima di passare alla applyTransformation --> rimossa perché non necessaria (rallenta esecuzione)

    // Copia la CDF dalla GPU alla CPU (più veloce grazie alla Pinned Memory) [senza sincronizzazione]
    cudaMemcpyAsync(h_cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Aspetta che la CDF sia copiata sulla CPU prima di procedere con la lookup table
    cudaDeviceSynchronize(); // Questa sincronizzazione è necessaria per evitare che la lookup table venga calcolata prima che h_cdf sia pronto

    // Calcolo della lookup table sulla CPU (Normalizzazione della CDF per ottenere i nuovi valori dei pixel)
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
    std::cout << "--> Tempo di esecuzione solo dei kernel CUDA: " << kernel_time << " ms" << std::endl;

    // Copia il risultato sulla CPU
    cudaMemcpy(output.data, d_output, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost); // Questa operazione è quella che rallenta l'esecuzione

    // Cleanup
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



