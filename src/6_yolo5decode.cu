
#include "xt_cuda.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#define num_threads   512

using namespace cv;
using namespace std;


struct AffineMatrix{
    float value[6];
};

struct Anchor {
    float width[3], height[3];

    Anchor(
        float w1, float h1,
        float w2, float h2,
        float w3, float h3
    ){
		width[0] = w1;
		width[1] = w2;
		width[2] = w3;
		height[0] = h1;
		height[1] = h2;
		height[2] = h3;	
	}
};

struct Box{
    float left, top, right, bottom;
    float confidence;
    int classes_id;
    unsigned int position;

    operator cv::Rect(){
        return cv::Rect(cv::Point(left, top), cv::Point(right, bottom));
    }
};

static vector<float> loadbinary(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<float> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length / sizeof(float));

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

static __device__ float sigmoid(float value) {
    return 1 / (1 + exp(-value));
}

static __device__ void project_affine(const AffineMatrix& H, float x, float y, float* u, float* v){
    float m_x1 = H.value[0];
    float m_y1 = H.value[1];
    float m_z1 = H.value[2];
    float m_x2 = H.value[3];
    float m_y2 = H.value[4];
    float m_z2 = H.value[5];
    *u = (x * m_x1 + y * m_y1 + m_z1);
    *v = (x * m_x2 + y * m_y2 + m_z2);
}

static __host__ float desigmoid(float val) {
    return -log(1 / val - 1);
}

static __global__ void decode_yolov5_kernel(float* data,
    int width, int height, int stride, float threshold, float threshold_desigmoid, int num_classes,
    Anchor anchor, Box* output, int* counter, int area, int maxobjs, AffineMatrix H, int edge) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= edge) return;

    int inner_offset = position % area;
    int a = position / area;
    float* ptr = data + (a * (num_classes + 5) + 4) * area + inner_offset;

    if (*ptr < threshold_desigmoid)
        return;

    float obj_confidence = sigmoid(*ptr);
    float* pclasses = ptr + area;
    float max_class_confidence = *pclasses;
    int max_classes = 0;
    pclasses += area;

    for (int j = 1; j < num_classes; ++j, pclasses += area) {
        if (*pclasses > max_class_confidence) {
            max_classes = j;
            max_class_confidence = *pclasses;
        }
    }

    max_class_confidence = sigmoid(max_class_confidence) * obj_confidence;
    if (max_class_confidence < threshold)
        return;

    int index = atomicAdd(counter, 1);
    if (index >= maxobjs)
        return;

    float* pbbox = ptr - 4 * area;
    float dx = sigmoid(*pbbox);  pbbox += area;
    float dy = sigmoid(*pbbox);  pbbox += area;
    float dw = sigmoid(*pbbox);  pbbox += area;
    float dh = sigmoid(*pbbox);  pbbox += area;

    int cell_x = position % width;
    int cell_y = (position / width) % height;
    float cx = (dx * 2 - 0.5f + cell_x) * stride;
    float cy = (dy * 2 - 0.5f + cell_y) * stride;
    float w = pow(dw * 2, 2) * anchor.width[a];
    float h = pow(dh * 2, 2) * anchor.height[a];
    float x = cx - w * 0.5f;
    float y = cy - h * 0.5f;
    float r = cx + w * 0.5f;
    float b = cy + h * 0.5f;
    Box& box = output[index];

    project_affine(H, x, y, &box.left, &box.top);
    project_affine(H, r, b, &box.right, &box.bottom);
    box.classes_id = max_classes;
    box.confidence = max_class_confidence;
    box.position = position;
}

static __device__ float boxIou(Box* a, Box* b){

    float cleft 	= max(a->left, b->left);
    float ctop 		= max(a->top, b->top);
    float cright 	= min(a->right, b->right);
    float cbottom 	= min(a->bottom, b->bottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = (a->right - a->left) * (a->bottom - a->top);
    float b_area = (b->right - b->left) * (b->bottom - b->top);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(void* bboxes, float threshold){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = *(int*)bboxes;
    if (position >= count) 
        return;

    Box* box_ptr = (Box*)((int*)bboxes + 1);
    Box* ptr_current = box_ptr + position;
    for(int i = 0; i < count; ++i){
        Box* a = ptr_current;
        Box* b = box_ptr + i;
        if(a == b || b->classes_id != a->classes_id) continue;

        if(b->confidence > a->confidence){
            float iou = boxIou(a, b);
            if(iou > threshold){
                // 如果发现iou大，并且b > a，置信度。b是第i个框，a是当前框
                // 表示当前框要过滤掉，不需要保留了
                a->classes_id = -1;
                return;
            }
        }
    }
} 

void yolo5decode(){

    Mat image = imread("bus.jpg");
    auto p8 = loadbinary("p8.binary");
    auto p16 = loadbinary("p16.binary");
    auto p32 = loadbinary("p32.binary");
    printf("p8.size = %d, 20 * 20 * 255 = %d\n", p8.size(), 80 * 80 * 255);
    printf("p16.size = %d, 40 * 40 * 255 = %d\n", p16.size(), 40 * 40 * 255);
    printf("p32.size = %d, 80 * 80 * 255 = %d\n", p32.size(), 20 * 20 * 255);

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* p8_device = nullptr;
    float* p16_device = nullptr;
    float* p32_device = nullptr;
    checkRuntime(cudaMalloc(&p8_device, p8.size() * sizeof(float)));
    checkRuntime(cudaMalloc(&p16_device, p16.size() * sizeof(float)));
    checkRuntime(cudaMalloc(&p32_device, p32.size() * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(p8_device,  p8.data(),  p8.size() *  sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(p16_device, p16.data(), p16.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(p32_device, p32.data(), p32.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    AffineMatrix matrix;
    Size affine_size(640, 640);

    // image长边缩放到affine长边大小
    float scale_factor = std::max(affine_size.width, affine_size.height) / (float)std::max(image.cols, image.rows);
    Mat scale = (Mat_<float>(3, 3) << 
        scale_factor, 0, 0,  
        0, scale_factor, 0,
        0, 0, 1
    );

    // image中心移动到affine中心
    Mat translation = (Mat_<float>(3, 3) << 
        1, 0, -image.cols * scale_factor * 0.5 + affine_size.width * 0.5,  
        0, 1, -image.rows * scale_factor * 0.5 + affine_size.height * 0.5,
        0, 0, 1
    );

    Mat m = translation * scale;
    Mat invert_m;
    cv::invertAffineTransform(m(Range(0, 2), Range(0, 3)), invert_m);
    memcpy(matrix.value, invert_m.ptr<float>(0), sizeof(float) * 6);
    
    int local_max_objs = 1000;
    size_t buffer_bytes_per_image = sizeof(Box) * local_max_objs + sizeof(int);
    char* box_array_device = nullptr;
    char* box_array_host   = new char[buffer_bytes_per_image];
    checkRuntime(cudaMalloc(&box_array_device, buffer_bytes_per_image));

    int num_classes = 80;
    float threshold = 0.25;
    float threshold_desigmoid = desigmoid(threshold);
    Box* box_ptr_device = (Box*)(box_array_device + sizeof(int));
    int* counter_ptr_device = (int*)box_array_device;
    vector<Anchor> anchors{
        {10.000000, 13.000000, 16.000000, 30.000000, 33.000000, 23.000000},
        {30.000000, 61.000000, 62.000000, 45.000000, 59.000000, 119.000000},
        {116.000000, 90.000000, 156.000000, 198.000000, 373.000000, 326.000000}
    };

    int tensor_sizes[] = {80, 40, 20};
    int strides[] = {8, 16, 32};
    float* tensor_ptrs[] = {p8_device, p16_device, p32_device};
    const int num_level = 3;

    for(int i = 0; i < num_level; ++i){
        int tensor_width = tensor_sizes[i];
        int tensor_height = tensor_sizes[i];
        int jobs = tensor_width * tensor_height * 3;
        int threads = jobs < num_threads ? jobs : num_threads;
        int blocks = ceil(jobs / (float)threads);
        checkKernel(decode_yolov5_kernel<<<blocks, threads, 0, nullptr>>>(tensor_ptrs[i], 
            tensor_width, tensor_height, strides[i], threshold, threshold_desigmoid, num_classes, anchors[i],
            box_ptr_device, counter_ptr_device, tensor_width * tensor_height, local_max_objs, matrix, jobs
        ));
    };

    int threads = local_max_objs < num_threads ? local_max_objs : num_threads;
    int blocks = ceil(local_max_objs / (float)threads);
    checkKernel(nms_kernel<<<blocks, threads, 0, nullptr>>>(box_array_device, 0.5));

    checkRuntime(cudaMemcpyAsync(box_array_host, box_array_device, buffer_bytes_per_image, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    Box* box_ptr_host = (Box*)(box_array_host + sizeof(int));
    int* counter_ptr_host = (int*)box_array_host;
    printf("Object count %d\n", *counter_ptr_host);

    int final_count = 0;
    for(int i = 0; i < *counter_ptr_host; ++i){
        if(box_ptr_host[i].classes_id != -1){
            cv::rectangle(image, box_ptr_host[i], Scalar(0, 255, 0), 2);
            final_count ++;
        }
    }

    printf("Final object count %d\n", final_count);
    imwrite("detect.jpg", image);
}