#include <hip/hip_runtime.h>
#include <stdio.h>

// Define a simple kernel to test global to shared memory loading
__global__ void globalToSharedTest(int* globalArray, int* outputArray, int size) {
    // Declare shared memory array
    extern __shared__ int sharedArray[];
    
    // Get thread and block indices
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int globalId = blockId * blockDim.x + threadId;
    
    // Load data from global to shared memory if within bounds
    if (threadId < size) {
        sharedArray[threadId] = globalArray[globalId];
    }
    
    // Synchronize threads to ensure all data is loaded
    __syncthreads();
    
    // Verify the loading by writing back to output array
    if (threadId < size) {
        outputArray[globalId] = sharedArray[threadId];
    }
}

int main() {
    const int ARRAY_SIZE = 256;
    const int BLOCK_SIZE = 64;
    const int GRID_SIZE = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int SHARED_MEM_SIZE = BLOCK_SIZE * sizeof(int);
    
    // Host arrays
    int* h_input = new int[ARRAY_SIZE];
    int* h_output = new int[ARRAY_SIZE];
    
    // Initialize input array with test values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = i;
    }
    
    // Device arrays
    int* d_input;
    int* d_output;
    
    // Allocate device memory
    hipMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    hipMalloc(&d_output, ARRAY_SIZE * sizeof(int));
    
    // Copy input data to device
    hipMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), hipMemcpyHostToDevice);
    
    // Launch kernel
    hipLaunchKernelGGL(globalToSharedTest, 
                      dim3(GRID_SIZE), 
                      dim3(BLOCK_SIZE), 
                      SHARED_MEM_SIZE,    // Dynamic shared memory size
                      0,                  // Stream
                      d_input, 
                      d_output, 
                      BLOCK_SIZE);
    
    // Copy results back to host
    hipMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    
    // Verify results
    bool testPassed = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (h_input[i] != h_output[i]) {
            printf("Test failed at index %d: expected %d, got %d\n", 
                   i, h_input[i], h_output[i]);
            testPassed = false;
            break;
        }
    }
    
    if (testPassed) {
        printf("Test passed! Global to shared memory loading works correctly.\n");
    }
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}

