
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <stdio.h>

// #define cublasCheck(op)		\
// 	{ auto status = (op); if(status != CUBLAS_STATUS_SUCCESS){ printf("cublas failure[%d]\n", status); abort(); }  }while(0);


// void transpose_gemm(cublasHandle_t handle, float* a, int arow, int acol, float* b, int brow, int bcol, int* ptr_crow, int* ptr_ccol, float* c) {

//     // 由于存在转置，所以这里行列调换
//     int opa_row = acol;
//     int opa_col = arow;
//     int opb_row = bcol;
//     int opb_col = brow;

//     int crow = opa_row;
//     int ccol = opb_col;
//     *ptr_crow = crow;
//     *ptr_ccol = ccol;

//     float alpha = 1;
//     float beta = 0;

//     /**
//      *   对于a、b、c都是列优先
//      *   对于 C = α op(A) op(B) + βC
//      *   op(A)应该是 m x k
//      *   op(B)应该是 k x n
//      *   C 应该是    m x n
//      * 
//      *   注意这里的m，是经过op后的m
//      *   这里假设A矩阵是(4x3)，所以m = 4, k = 3
//      *   这里假设B矩阵是(3x2)，所以k = 3, n = 2
//      *   也因此C是(m x n, 4x2), m = 4, n = 2
//      *   
//      *   注意这里的lda取值为A矩阵的移动到下一列偏移量，按照列优先，则取其A行数为偏移到下一列的量。因此是A.row = 4
//      *   注意这里的ldb取值为B矩阵的移动到下一列偏移量，按照列优先，则取其B行数为偏移到下一列的量。因此是B.row = 3
//      *   注意这里的ldc取值为C矩阵的移动到下一列偏移量，按照列优先，则取其C行数为偏移到下一列的量。因此是C.row = 4
//      * 
//      *   CUBLAS_OP_N   为  A = op(A)     identity
//      *   CUBLAS_OP_T   为  A.T = op(A)   转置
//      *   CUBLAS_OP_C   为  A.H = op(A)   共轭转置
//     **/
//     cublasCheck(cublasSgemm(
//         handle,            //  cublasHandle_t handle, 
//         CUBLAS_OP_T,       //  cublasOperation_t transa, 
//         CUBLAS_OP_T,       //  cublasOperation_t transb,   
//         opa_row,              //  int m,          op(A).row
//         opb_col,              //  int n,          op(B).col
//         opa_col,              //  int k,          op(A).col  or  op(B).row
//         &alpha,            //  const float *alpha, /* host or device pointer */  
//         a,                 //  const float *A, 
//         arow,              //  int lda,         A.row
//         b,                 //  const float *B,
//         brow,              //  int ldb,         B.row
//         &beta,             //  const float *beta, /* host or device pointer */  
//         c,                 //  float *C, 
//         crow               //  int ldc);        C.row 
//     ));
// }

// int main(){

//     // 按照列优先，其表示的矩阵3x4为
//     // 0  1  3  3
//     // 1  0  1  0
//     // 2  1  2  0
//     float a_host[] = {
//         0, 1, 2, 1, 0, 1, 3, 1, 2, 3, 0, 0
//     };

//     // 按照列优先，其表示的矩阵2x3为
//     // 2  3  0
//     // 1  2  1
//     float b_host[] = {
//         2, 1, 3, 2, 0, 1
//     };

//     float c_host[8] = {0};

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
    
//     int arow = 3;
//     int acol = 4;
//     int brow = 2;
//     int bcol = 3;
//     int crow = 0;
//     int ccol = 0;

//     // 计算 C = A.T @ B.T
//     //      C = (4x3) @ (3x2)
//     //      C = 4x2
//     transpose_gemm(cublas_handle, a_device, arow, acol, b_device, brow, bcol, &crow, &ccol, c_device);

//     cudaMemcpy(c_host, c_device, sizeof(c_host), cudaMemcpyDeviceToHost);
//     for(int p =0 ; p < crow * ccol; ++p){
//         printf("%f ", c_host[p]);
//     }
//     printf("\n");

//     printf("打印成人类可观察的样子：\n");
//     for(int index_row = 0; index_row < crow; ++index_row){
//         for(int index_col = 0; index_col < ccol; ++index_col){
//             // 一行一行的打印
//             int p = index_row + index_col * crow;
//             printf("%f ", c_host[p]);
//         }    
//         printf("\n");
//     }
//     printf("\n");
//     return 0;
// }