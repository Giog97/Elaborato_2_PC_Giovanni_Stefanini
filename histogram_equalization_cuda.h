//
// Created by giost on 07/02/2025.
//

#ifndef KERNEL_H
#define KERNEL_H

#include <opencv2/opencv.hpp>

// Dichiarazione delle funzioni CUDA
void histogram_equalization_cuda(const cv::Mat& input, cv::Mat& output);

// Dichiarazione dei kernel CUDA
__global__ void computeHistogram(const uchar* input, int* hist, int width, int height);
__global__ void computeCDF(int* hist, int* cdf);
__global__ void applyTransformation(uchar* output, const uchar* input, const uchar* lookup_table, int width, int height);

#endif // KERNEL_H
