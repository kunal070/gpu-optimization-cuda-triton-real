/**
 * Header file for basic CUDA kernels
 * Provides function declarations for vector addition and matrix multiplication
 */

#ifndef BASIC_KERNELS_H
#define BASIC_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Vector addition: c = a + b
void vector_add_cuda(
    const float* a,
    const float* b,
    float* c,
    int n
);

// Matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
void matrix_multiply_cuda(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    bool use_tiled = true
);

#ifdef __cplusplus
}
#endif

#endif // BASIC_KERNELS_H

