//
// Created by giost on 10/02/2025.
//

#include "histogram_equalization_seq.h"
#include <algorithm> // Per min_element
#include <opencv2/opencv.hpp>
using namespace cv;
using std::min_element;

// Funzione per l'equalizzazione dell'istogramma
void histogram_equalization_seq(const Mat& input, Mat& output) {
    // Converti l'immagine in scala di grigi se non lo è già
    if (input.channels() > 1) {
        cvtColor(input, output, COLOR_BGR2GRAY);
    } else {
        output = input.clone();
    }

    // 1. Calcolo dell'istogramma (256 livelli di intensità)
    int hist[256] = {0};
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            hist[output.at<uchar>(y, x)]++;
        }
    }

    // 2. Calcolo della funzione di distribuzione cumulativa (CDF)
    int cdf[256] = {0};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // 3. Normalizzazione della CDF per ottenere i nuovi valori dei pixel
    int total_pixels = output.rows * output.cols;
    int min_cdf = *min_element(cdf, cdf + 256); // Per evitare valori nulli

    uchar lookup_table[256];
    for (int i = 0; i < 256; i++) {
        lookup_table[i] = (uchar)(((cdf[i] - min_cdf) * 255) / (total_pixels - min_cdf));
    }

    // 4. Applica la trasformazione all'immagine
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            output.at<uchar>(y, x) = lookup_table[output.at<uchar>(y, x)];
        }
    }
}