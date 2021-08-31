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
//     // 计算矩阵乘法，输入的A、B是行优先，以及其维度也是行优先的方式描述的。他的所有描述都是符合行优先习惯
//     // 计算 C = (AB).T  
//     // OP_N  -> identity
//     // float a_host[] = {
//     //     0, 1, 2, 1, 0, 1, 3, 1, 2, 3, 0, 0
//     // };
//     // arow = 4,  acol = 3
//     // 准备4x3矩阵A
//     // float a_host[] = {
//     //     0, 1, 2, 
//     //     1, 0, 1, 
//     //     3, 1, 2, 
//     //     3, 0, 0
//     // };
//     // 用cublas的眼睛去看A矩阵，会等价什么样子
//     // 定义了矩阵 3x4  
//     // 0  1  3  3
//     // 1  0  1  0
//     // 2  1  2  0
//     // 我用cublas眼光看矩阵A，然后对矩阵A做转置A.T。得到的就是行优先视角下想描述表达的东西
//     // cublas_a_row = acol
//     // cublas_a_col = arow
//     // op_a_row = cublas_a_col = arow
//     // op_a_col = cublas_a_row = acol

//     int op_a_row = arow;
//     int op_a_col = acol;
//     int op_b_row = brow;
//     int op_b_col = bcol;
//     int crow = op_a_row;
//     int ccol = op_b_col;
//     int cublas_a_row = acol;
//     int cublas_b_row = bcol;
//     int cublas_c_row = crow;
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
//         cublas_a_row,          //lda  移动到下一列所需要的偏移量
//         b,             //B
//         cublas_b_row,          //ldb  移动到下一列所需要的偏移量
//         &beta,         //beta
//         c,             //C
//         cublas_c_row           //ldc  移动到下一列所需要的偏移量
//     ));
// }

// int main(){

//     // 准备4x3矩阵A
//     float a_host[] = {
//         0, 1, 2, 
//         1, 0, 1, 
//         3, 1, 2, 
//         3, 0, 0
//     };

//     // 准备3x2矩阵B
//     float b_host[] = {
//         2, 1, 
//         3, 2,
//         0, 1
//     };

//     // 准备4x2矩阵C
//     float c_host[8] = {0};

//     int arow = 4;
//     int acol = 3;
//     int brow = 3;
//     int bcol = 2;
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