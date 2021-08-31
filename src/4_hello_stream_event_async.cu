
#include "xt_cuda.hpp"
#include <math.h>

#define num_threads   256

static __global__ void compute(int* input, int* output, int edge){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= edge)
        return;

    output[index] = input[index] * 5;
}

void hello_stream_event_async(){

    int size = 100000;
    int bytes = size * sizeof(int);
    int* ihost = new int[size];
    int* ohost = new int[size];
    int* idev = nullptr;
    int* odev = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t  memcpy_recorder_start = nullptr, memcpy_recorder_stop = nullptr;
    cudaEvent_t  kernel_recorder_start = nullptr, kernel_recorder_stop = nullptr;

    checkRuntime(cudaStreamCreate(&stream));
    checkRuntime(cudaEventCreate(&memcpy_recorder_start));
    checkRuntime(cudaEventCreate(&memcpy_recorder_stop));
    checkRuntime(cudaEventCreate(&kernel_recorder_start));
    checkRuntime(cudaEventCreate(&kernel_recorder_stop));

    for(int i = 0; i < size; ++i)
        ihost[i] = i + 1;

    checkRuntime(cudaMalloc(&idev, bytes));
    checkRuntime(cudaMalloc(&odev, bytes));
    
    checkRuntime(cudaEventRecord(memcpy_recorder_start, stream));

    int n_repeat = 1000;
    for(int i = 0; i < n_repeat; ++i)
        checkRuntime(cudaMemcpyAsync(idev, ihost, bytes, cudaMemcpyHostToDevice, stream));

    checkRuntime(cudaEventRecord(memcpy_recorder_stop, stream));

    int threads = size < num_threads ? size : num_threads;
    int blocks = ceil(size / (float)threads);
    printf("blocks = %d, threads = %d\n", blocks, threads);

    checkRuntime(cudaEventRecord(kernel_recorder_start, stream));

    for(int i = 0; i < n_repeat; ++i)
        checkKernel(compute<<<blocks, threads, 0, stream>>>(idev, odev, size));

    checkRuntime(cudaEventRecord(kernel_recorder_stop, stream));
    checkRuntime(cudaMemcpyAsync(ohost, odev, bytes, cudaMemcpyDeviceToHost, stream));

    printf("Input: ");
    for(int i = 0; i < 100; ++i)
        printf("%d ", ihost[i]);
    printf("\n");

    printf("Output: ");
    for(int i = 0; i < 100; ++i)
        printf("%d ", ohost[i]);
    printf("\n");

    float time_memcpy, time_kernel;
    checkRuntime(cudaEventElapsedTime(&time_memcpy, memcpy_recorder_start, memcpy_recorder_stop));
    checkRuntime(cudaEventElapsedTime(&time_kernel, kernel_recorder_start, kernel_recorder_stop));

    printf("Repeat[%d] Memcpy %.5f ms, total memory %.2f MB\n", n_repeat, time_memcpy, bytes * n_repeat / 1024.0f / 1024.0f);
    printf("Repeat[%d] Kernel %.5f ms\n", n_repeat, time_kernel);
}