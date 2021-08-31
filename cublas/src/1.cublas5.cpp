
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <stdio.h>

// #define cublasCheck(op)		\
// 	{ auto status = (op); if(status != CUBLAS_STATUS_SUCCESS){ printf("cublas failure[%d]\n", status); abort(); }  }while(0);

// // 输入为行排列，矩阵描述也是行排列。此时需要运算得到C，是列排列（等价于转置）
// // 计算 C = (AB).T，此时A和B和C都是行排列内存
// void gemm(cublasHandle_t handle, float* a, int arow, int acol, float* b, int brow, int bcol, int* ptr_crow, int* ptr_ccol, float* c) {

//     // 由于提供的数据行排列，因此描述的矩阵其实是转置的矩阵
//     // C语言描述是1x3
//     // cublas描述的矩阵是3x1
//     // m = {
//     //    1, 2, 3
//     // }
//     // 因此_arow = acol;   _acol = arow;
//     int _arow = acol;
//     int _acol = arow;
//     int _brow = bcol;
//     int _bcol = brow;

//     // 由于存在转置，所以这里行列调换
//     int opa_row = _acol;
//     int opa_col = _arow;
//     int opb_row = _bcol;
//     int opb_col = _brow;

//     int crow = opa_row;
//     int ccol = opb_col;

//     // 由于返回的数据是列排列，外面解释为行排列，因此返回的shape做转置
//     *ptr_crow = ccol;
//     *ptr_ccol = crow;

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
//         _arow,              //  int lda,         A.row
//         b,                 //  const float *B,
//         _brow,              //  int ldb,         B.row
//         &beta,             //  const float *beta, /* host or device pointer */  
//         c,                 //  float *C, 
//         crow               //  int ldc);        C.row 
//     ));
// }

// int main(){

//     float a_host[] = {
//         0, 1, 2, 
//         1, 0, 1, 
//         3, 1, 2, 
//         3, 0, 0
//     };

//     float b_host[] = {
//         2, 1, 
//         3, 2, 
//         0, 1
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
    
//     int arow = 4;
//     int acol = 3;
//     int brow = 3;
//     int bcol = 2;
//     int crow = 0;
//     int ccol = 0;

//     // 计算 C = (AB).T
//     //      C = ((4x3) @ (3x2)).T
//     //      C = 2x4
//     gemm(cublas_handle, a_device, arow, acol, b_device, brow, bcol, &crow, &ccol, c_device);

//     cudaMemcpy(c_host, c_device, sizeof(c_host), cudaMemcpyDeviceToHost);
//     for(int p =0 ; p < crow * ccol; ++p){
//         printf("%f ", c_host[p]);
//     }
//     printf("\n");

//     printf("打印成人类可观察的样子，这里按照行排列打印：\n");
//     for(int index_row = 0; index_row < crow; ++index_row){
//         for(int index_col = 0; index_col < ccol; ++index_col){
//             // 一行一行的打印
//             int p = index_row * ccol + index_col;
//             printf("%f ", c_host[p]);
//         }    
//         printf("\n");
//     }
//     printf("\n");
//     return 0;
// }