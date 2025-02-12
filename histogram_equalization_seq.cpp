//
// Created by giost on 05/02/2025.
//

#include "histogram_equalization_seq.h"
#include <algorithm> // Per min_element
#include <opencv2/opencv.hpp>
using namespace cv;
using std::min_element;

// Funzione per l'equalizzazione dell'istogramma
void histogram_equalization_seq(const Mat& input, Mat& output) {
    // Converti l'immagine in scala di grigi se non lo è già
    if (input.channels() > 1) { // Controllo sul numero di canali (se + di 1, allora immagine a colori)
        cvtColor(input, output, COLOR_BGR2GRAY);
    } else {
        output = input.clone(); // Se immagine è grigia la clono per non modificarla direttamente
    }

    // 1. Calcolo dell'istogramma (256 livelli di intensità) //NB doppio ciclo
    int hist[256] = {0}; // Istogramma memorizzato in array di 256 elementi, ogni posizione corrisponde a numero di pixel di quell’intensità.
    // Codice scansiona l’intera immagine e conta quanti pixel hanno un determinato valore di luminosità
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            hist[output.at<uchar>(y, x)]++;
        }
    }

    // 2. Calcolo della funzione di distribuzione cumulativa (CDF)
    int cdf[256] = {0};
    cdf[0] = hist[0]; // La CDF permette di capire quanti pixel sono distribuiti sotto un certo livello di intensità
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i]; // Funzione costruita sommando progressivamente valori dell’istogramma
    }

    // 3. Normalizzazione della CDF per ottenere i nuovi valori dei pixel (serve per garantire che i valori stiano tra 0 e 255)
    int total_pixels = output.rows * output.cols;
    int min_cdf = *min_element(cdf, cdf + 256); // Per evitare valori nulli prende primo valore non nullo della CDF

    uchar lookup_table[256]; // Creo una tabella che assegna a ogni valore di intensità originale un nuovo valore equalizzato.
    for (int i = 0; i < 256; i++) {
        lookup_table[i] = (uchar)(((cdf[i] - min_cdf) * 255) / (total_pixels - min_cdf));
    }

    // 4. Applica la trasformazione all'immagine //NB doppio ciclo
    for (int y = 0; y < output.rows; y++) {
        for (int x = 0; x < output.cols; x++) {
            output.at<uchar>(y, x) = lookup_table[output.at<uchar>(y, x)]; // Sostituisce i vecchi valori dei pixel con quelli equalizzati
        }
    }
}