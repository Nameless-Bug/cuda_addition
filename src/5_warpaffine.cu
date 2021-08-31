
#include "xt_cuda.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define num_threads   512

using namespace cv;
using namespace std;


struct AffineMatrix{
    float value[6];
};

static __global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height,
	AffineMatrix matrix_2_3, uint8_t const_value_st, int edge
){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = matrix_2_3.value[0];
    float m_y1 = matrix_2_3.value[1];
    float m_z1 = matrix_2_3.value[2];
    float m_x2 = matrix_2_3.value[3];
    float m_y2 = matrix_2_3.value[4];
    float m_z2 = matrix_2_3.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = (m_x1 * dx + m_y1 * dy + m_z1) + 0.5f;
    float src_y = (m_x2 * dx + m_y2 * dy + m_z2) + 0.5f;

    if(src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height){
        uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
        *pdst++ = const_value_st; *pdst++ = const_value_st; *pdst++ = const_value_st;
        return;
    }

    int y_low = floor(src_y);
    int x_low = floor(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;

    uint8_t* v1 = const_value;
    uint8_t* v2 = const_value;
    uint8_t* v3 = const_value;
    uint8_t* v4 = const_value;
    if(y_low >= 0){
        if (x_low >= 0)
            v1 = src + y_low * src_line_size + x_low * 3;

        if (x_high < src_width)
            v2 = src + y_low * src_line_size + x_high * 3;
    }
    
    if(y_high < src_height){
        if (x_low >= 0)
            v3 = src + y_high * src_line_size + x_low * 3;

        if (x_high < src_width)
            v4 = src + y_high * src_line_size + x_high * 3;
    }

    #pragma unroll 3
    for(int i = 0 ; i < 3; ++i)
        *pdst++ = (w1 * *v1++ + w2 * *v2++ + w3 * *v3++ + w4 * *v4++ + 0.5f);
}

void warpaffine(){

    Mat image = imread("bus.jpg");
    size_t image_bytes = image.cols * image.rows * 3;
    uint8_t* image_device = nullptr;

    Mat affine(640, 640, CV_8UC3);
    size_t affine_bytes = affine.cols * affine.rows * 3;
    uint8_t* affine_device = nullptr;

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    checkRuntime(cudaMalloc(&image_device, image_bytes));
    checkRuntime(cudaMalloc(&affine_device, affine_bytes));
    checkRuntime(cudaMemcpyAsync(image_device, image.data, image_bytes, cudaMemcpyHostToDevice, stream));

    int jobs = affine.cols * affine.rows;
    int threads = jobs < num_threads ? jobs : num_threads;
    int blocks = ceil(jobs / (float)threads);
    printf("blocks = %d, threads = %d\n", blocks, threads);

    AffineMatrix matrix;

    // image长边缩放到affine长边大小
    float scale_factor = std::max(affine.cols, affine.rows) / (float)std::max(image.cols, image.rows);
    Mat scale = (Mat_<float>(3, 3) << 
        scale_factor, 0, 0,  
        0, scale_factor, 0,
        0, 0, 1
    );

    // image中心移动到affine中心
    Mat translation = (Mat_<float>(3, 3) << 
        1, 0, -image.cols * scale_factor * 0.5 + affine.cols * 0.5,  
        0, 1, -image.rows * scale_factor * 0.5 + affine.rows * 0.5,
        0, 0, 1
    );

    Mat m = translation * scale;
    Mat invert_m;
    cv::invertAffineTransform(m(Range(0, 2), Range(0, 3)), invert_m);
    memcpy(matrix.value, invert_m.ptr<float>(0), sizeof(float) * 6);

    warp_affine_bilinear_kernel<<<blocks, threads, 0, stream>>>(
        image_device, image.cols * 3, image.cols, image.rows,
        affine_device, affine.cols * 3, affine.cols, affine.rows,
        matrix, 114, jobs
    );

    checkRuntime(cudaMemcpyAsync(affine.data, affine_device, affine_bytes, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    imwrite("affine.jpg", affine);
}