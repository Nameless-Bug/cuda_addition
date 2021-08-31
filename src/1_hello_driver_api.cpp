
#include "xt_cuda.hpp"

void hello_driver_api(){
    CUcontext ctx = nullptr;
    CUdeviceptr ptr = 0;
    CUdevice device = 1;

    checkDriver(cuInit(0));
    //checkDriver(cuDevicePrimaryCtxRetain(&ctx, 0));
    checkDriver(cuCtxCreate(&ctx, 0, device));
    checkDriver(cuCtxPushCurrent(ctx));
    checkDriver(cuMemAlloc(&ptr, 32));
    printf("MemAlloc ptr = %p\n", (void*)ptr);

    int device_count = 0;
    checkDriver(cuDeviceGetCount(&device_count));
    printf("Device count is %d\n", device_count);

    char name[100];
    checkDriver(cuDeviceGetName(name, sizeof(name), device));
    printf("Device[%d] name is %s\n", device, name);

    size_t total_mem = 0;
    checkDriver(cuDeviceTotalMem(&total_mem, device));
    printf("Device[%d] total mem is %.2f GB\n", device, total_mem / 1024.0f / 1024.0f / 1024.0f);

    int value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 0;
    checkDriver(cuDeviceGetAttribute(&value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
    printf("Device[%d] MAX_SHARED_MEMORY_PER_MULTIPROCESSOR is %.2f KB\n", device, value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR / 1024.0f);

    int value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 0;
    checkDriver(cuDeviceGetAttribute(&value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    printf("Device[%d] MAX_SHARED_MEMORY_PER_BLOCK is %.2f KB\n", device, value_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK / 1024.0f);

    int value_CU_DEVICE_ATTRIBUTE_WARP_SIZE = 0;
    checkDriver(cuDeviceGetAttribute(&value_CU_DEVICE_ATTRIBUTE_WARP_SIZE, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
    printf("Device[%d] WARP_SIZE is %d\n", device, value_CU_DEVICE_ATTRIBUTE_WARP_SIZE);

    int block_dim_x, block_dim_y, block_dim_z, grid_dim_x, grid_dim_y, grid_dim_z;
    checkDriver(cuDeviceGetAttribute(&block_dim_x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
    checkDriver(cuDeviceGetAttribute(&block_dim_y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
    checkDriver(cuDeviceGetAttribute(&block_dim_z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
    checkDriver(cuDeviceGetAttribute(&grid_dim_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    checkDriver(cuDeviceGetAttribute(&grid_dim_y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
    checkDriver(cuDeviceGetAttribute(&grid_dim_z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));
    printf("Device[%d] MAX_BLOCK_DIM is x=%d, y=%d, z=%d\n", device, block_dim_x, block_dim_y, block_dim_z);
    printf("Device[%d] MAX_GRID_DIM is x=%d, y=%d, z=%d\n", device, grid_dim_x, grid_dim_y, grid_dim_z);
}