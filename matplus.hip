#include <hip/hip_runtime.h>
#include <stdio.h>

// Kernel for matrix addition using shared memory
__global__ void matrixAdd(float *A, float *B, float *C, int width) {
    // Define shared memory (local memory in HIP)
    __shared__ float s_A[32][32];
    __shared__ float s_B[32][32];

    // Global indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data into shared memory if within matrix bounds
    if (gx < width && gy < width) {
        s_A[ty][tx] = A[gy * width + gx];
        s_B[ty][tx] = B[gy * width + gx];
    }

    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Perform addition and store result if within bounds
    if (gx < width && gy < width) {
        C[gy * width + gx] = s_A[ty][tx] + s_B[ty][tx];
    }
}

int main() {
    const int WIDTH = 1024;  // Matrix width (assuming square matrices)
    const int SIZE = WIDTH * WIDTH;
    const int BYTES = SIZE * sizeof(float);

    // Host matrices
    float *h_A = (float*)malloc(BYTES);
    float *h_B = (float*)malloc(BYTES);
    float *h_C = (float*)malloc(BYTES);

    // Initialize host matrices with sample data
    for (int i = 0; i < SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device matrices
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, BYTES);
    hipMalloc(&d_B, BYTES);
    hipMalloc(&d_C, BYTES);

    // Copy data to device
    hipMemcpy(d_A, h_A, BYTES, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, BYTES, hipMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32);  // 32x32 threads per block
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, 
                 (WIDTH + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);

    // Copy result back to host
    hipMemcpy(h_C, d_C, BYTES, hipMemcpyDeviceToHost);

    // Verify a sample element
    printf("C[0] = %f (should be 3.0)\n", h_C[0]);

    // Clean up
    free(h_A); free(h_B); free(h_C);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);

    return 0;
}

