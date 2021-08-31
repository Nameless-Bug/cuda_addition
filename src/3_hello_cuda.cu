
#include "xt_cuda.hpp"
#include <math.h>

#define num_threads   256

static __global__ void compute(int* input, int* output, int edge){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= edge)
        return;

    output[index] = input[index] * 5;
}

void hello_cuda(){

    int size = 100000;
    int bytes = size * sizeof(int);
    int* ihost = new int[size];
    int* ohost = new int[size];
    int* idev = nullptr;
    int* odev = nullptr;

    for(int i = 0; i < size; ++i)
        ihost[i] = i + 1;

    checkRuntime(cudaMalloc(&idev, bytes));
    checkRuntime(cudaMalloc(&odev, bytes));
    checkRuntime(cudaMemcpy(idev, ihost, bytes, cudaMemcpyHostToDevice));

    int threads = size < num_threads ? size : num_threads;
    int blocks = ceil(size / (float)threads);
    printf("blocks = %d, threads = %d\n", blocks, threads);

    checkKernel(compute<<<blocks, threads, 0, nullptr>>>(idev, odev, size));
    checkRuntime(cudaMemcpy(ohost, odev, bytes, cudaMemcpyDeviceToHost));

    printf("Input: ");
    for(int i = 0; i < 100; ++i)
        printf("%d ", ihost[i]);
    printf("\n");

    printf("Output: ");
    for(int i = 0; i < 100; ++i)
        printf("%d ", ohost[i]);
    printf("\n");
}