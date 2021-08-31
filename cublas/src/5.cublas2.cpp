// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <stdio.h>

// #define cublasCheck(op)                                 \
//     do{                                                 \
//         auto status = (op);                             \
//         if(status != CUBLAS_STATUS_SUCCESS){            \
//             printf("cublas failure[%d]\n", status);     \
//             abort();                                    \
//         }                                               \
//     }while(0);


// void gemm(cublasHandle_t handle, 
//     float* a, int arow, int acol, 
//     float* b, int brow, int bcol,
//     float* c, int* ptr_crow, int& ptr_ccol
// ){
//     // OP_N  -> identity
//     int op_a_row = acol;
//     int op_a_col = arow;
//     int op_b_row = bcol;
//     int op_b_col = brow;
//     int crow = op_a_row;
//     int ccol = op_b_col;
//     *ptr_crow = crow;
//     ptr_ccol = ccol;

//     float alpha = 1.0f;
//     float beta = 0.0f;
    
//     // C = alpha op(A) op(B) + beta C
//     // C = A.T @ B.T
//     //   A[4x3], B[3x2]
//     //   A[3x4] @ B[2x3]
//     // op(A).shape = m x k
//     // op(B).shape = k x n
//     // C.shape = m x n
//     cublasCheck(cublasSgemm(
//         handle,
//         CUBLAS_OP_T,   //transa    A做转置
//         CUBLAS_OP_T,   //transb    B做转置
//         op_a_row,      //m    op(A).row
//         op_b_col,      //n    op(B).col
//         op_a_col,      //k    op(A).col or op(B).row
//         &alpha,        //alpha
//         a,             //A
//         arow,          //lda  移动到下一列所需要的偏移量
//         b,             //B
//         brow,          //ldb  移动到下一列所需要的偏移量
//         &beta,         //beta
//         c,             //C
//         crow           //ldc  移动到下一列所需要的偏移量
//     ));
// }

// int main(){

//     // 准备3x4矩阵A
//     // 0  1  3  3 
//     // 1  0  1  0
//     // 2  1  2  0
//     float a_host[] = {
//         0, 1, 2, 1, 0, 1, 3, 1, 2, 3, 0, 0
//     };

//     // 准备2x3矩阵B
//     // 2  3  0
//     // 1  2  1
//     float b_host[] = {
//         2, 1, 3, 2, 0, 1
//     };

//     // 准备4x2矩阵C
//     float c_host[8] = {0};

//     int arow = 3;
//     int acol = 4;
//     int brow = 2;
//     int bcol = 3;
//     int crow = 0;
//     int ccol = 0; 

//     // 定义device内存，并进行复制
//     float* a_device = nullptr;
//     float* b_device = nullptr;
//     float* c_device = nullptr;
//     cublasHandle_t cublas_handle = nullptr;
//     cublasCreate(&cublas_handle);

//     cudaMalloc(&a_device, sizeof(a_host));
//     cudaMalloc(&b_device, sizeof(b_host));
//     cudaMalloc(&c_device, sizeof(c_host));

//     cudaMemcpy(a_device, a_host, sizeof(a_host), cudaMemcpyHostToDevice);
//     cudaMemcpy(b_device, b_host, sizeof(b_host), cudaMemcpyHostToDevice);

//     gemm(
//         cublas_handle,
//         a_device,
//         arow,
//         acol,
//         b_device,
//         brow,
//         bcol,
//         c_device,
//         &crow,
//         ccol
//     );

//     cudaMemcpy(c_host, c_device, sizeof(c_host), cudaMemcpyDeviceToHost);

//     for(int p = 0; p < crow * ccol; ++p){
//         printf("%f ", c_host[p]);
//     }
//     printf("\n");

//     printf("按照人类习惯打印\n");
//     for(int row_index = 0; row_index < crow; ++row_index){
//         for(int col_index = 0; col_index < ccol; ++col_index){
//             // row -> 0, col -> 0
//             // p = 0
//             // row -> 1, col -> 1
//             // crow = 4,   ccol = 2
//             // p = row + crow * col
//             int p = row_index + crow * col_index;
//             printf("%f ", c_host[p]);
//         }
//         printf("\n");
//     }
//     return 0;
// }