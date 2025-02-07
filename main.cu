//
// Created by giost on 05/02/2025.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <chrono> // Libreria per il timing
#include "kernel.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

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

int main() {
    string img_dir = "img";  // Nome della cartella con le immagini
    string result_dir = "img_results";  // Nome della cartella per i risultati

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // Serve per avere un output più leggibile (sennò OpenCV manda dei log in output)

    // Verifica se la cartella delle immagini esiste
    if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) {
        cerr << "Errore: la cartella '" << img_dir << "' non esiste!" << endl;
        return -1;
    }

    // Crea la cartella per i risultati se non esiste
    if (!fs::exists(result_dir)) {
        fs::create_directory(result_dir);
    }

    // Lista dei file disponibili nella cartella
    vector<string> images;
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.is_regular_file()) {
            images.push_back(entry.path().string());
        }
    }

    // Se non ci sono immagini, uscire
    if (images.empty()) {
        cerr << "Errore: nessuna immagine trovata nella cartella '" << img_dir << "'!" << endl;
        return -1;
    }

    // Mostra la lista di immagini disponibili
    cout << "Scegli un'immagine da equalizzare:\n";
    for (size_t i = 0; i < images.size(); ++i) {
        cout << i + 1 << ") " << images[i] << endl;
    }

    // Scegli un numero
    size_t choice;
    cout << "Inserisci il numero dell'immagine: ";
    cin >> choice;

    if (choice < 1 || choice > images.size()) {
        cerr << "Errore: scelta non valida!" << endl;
        return -1;
    }

    string selected_image = images[choice - 1];

    // Carica l'immagine
    Mat input = imread(selected_image, IMREAD_COLOR);
    if (input.empty()) {
        cout << "Errore: impossibile caricare l'immagine!" << endl;
        return -1;
    }

    // Converti l'immagine in scala di grigi
    Mat input_gray;
    cvtColor(input, input_gray, COLOR_BGR2GRAY);

    // Output
    Mat output_seq;

    // Inizializza output_cuda con le stesse dimensioni e tipo di input_gray
    Mat output_cuda = Mat::zeros(input_gray.size(), input_gray.type());

    // *** MISURAZIONE TEMPO DI ESECUZIONE SEQUENZIALE ***
    auto start_seq = chrono::high_resolution_clock::now(); // Tempo iniziale
    histogram_equalization_seq(input_gray, output_seq);
    auto end_seq = chrono::high_resolution_clock::now(); // Tempo finale
    chrono::duration<double, milli> elapsed_seq = end_seq - start_seq;  // Calcola differenza in millisecondi

    cout << "Tempo di esecuzione dell'algoritmo sequenziale: " << elapsed_seq.count() << " ms" << endl;

    // *** MISURAZIONE TEMPO DI ESECUZIONE CUDA ***
    auto start_cuda = chrono::high_resolution_clock::now(); // Tempo iniziale
    histogram_equalization_cuda(input_gray, output_cuda);
    auto end_cuda = chrono::high_resolution_clock::now(); // Tempo finale
    chrono::duration<double, milli> elapsed_cuda = end_cuda - start_cuda; // Calcola differenza in millisecondi

    cout << "Tempo di esecuzione CUDA (kernel + overhead di comunicazione tra CPU e GPU): " << elapsed_cuda.count() << " ms" << endl;

    // Salva i risultati - Costruisci il percorso completo per il salvataggio dell'immagine
    string output_path_seq = result_dir + "/equalized_seq_" + fs::path(selected_image).filename().string();
    string output_path_cuda = result_dir + "/equalized_cuda_" + fs::path(selected_image).filename().string();

    // Salva il risultato nella cartella img_results
    imwrite(output_path_seq, output_seq);
    imwrite(output_path_cuda, output_cuda);

    cout << "Immagine sequenziale salvata come: " << output_path_seq << endl;
    cout << "Immagine CUDA salvata come: " << output_path_cuda << endl;

    return 0;
}

