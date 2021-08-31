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


//     /*
//     cublasSgemm 函数，实现单精度通用矩阵乘法
//     参数：
//     handle,       cublas的句柄
//     transa,       对a的操作方式，有OP_N, OP_T, OP_C
//     transb,       对b的操作方式，有OP_N, OP_T, OP_C
//     m,            op(A).row
//     n,            op(B).col
//     k,            op(A).col or op(B).row
//     alpha,        
//     a pointer,
//     lda,          对于A矩阵，获取其下一列所需要的偏移数
//     b pointer,
//     ldb,          对于B矩阵，获取其下一列所需要的偏移数
//     beta,
//     c pointer,
//     ldc           对于C矩阵，获取其下一列所需要的偏移数

//     描述：
//     实现 C = alpha x op(A) op(B) + beta C
//         op(A).shape  = m x k
//         op(B).shape  = k x n
//         C.shape      = m x n
//     对于a pointer, b pointer, c pointer，提供的数据以列优先排列

//     对于cublasOperation_t，有三种：
//     CUBLAS_OP_N  :  identity，  op(A) = A
//     CUBLAS_OP_T  :  转置,       op(A) = A^T   transpose
//     CUBLAS_OP_C  :  共轭转置,    op(A) = A^H   conj

//     对于具体实现，第一种实现方式，CUBLAS_OP_N，实现矩阵乘法
//     定义矩阵A(4x3)
//     0  1  2
//     1  0  1
//     3  1  2
//     3  0  0

//     由于是列优先，因此写下来如下：
//     A = {0, 1, 3, 3, 1, 0, 1, 0, 2, 1, 2, 0}

//     定义矩阵B(3x2)
//     2  1
//     3  2
//     0  1
//     B = {2, 3, 0, 1, 2, 1}

//     计算矩阵乘法 C = AB,  C.shape = 4x2
//     cublasSgemm(
//         handle,
//         transa = CUBLAS_OP_N, transb = CUBLAS_OP_N, 
//         m = op(A).row = 4,         
//         n = op(B).col = 2,
//         k = op(A).col or op(B).row = 3,            
//         alpha,
//         a pointer,
//         lda = A.row = 4,
//         b pointer,
//         ldb = B.row = 3,
//         beta,
//         c pointer,
//         ldc = C.row = 4
//     )
//     计算结果为：3, 2, 9, 6, 4, 2, 7, 3，转换为4x2，列优先为：
//     3  4
//     2  2
//     9  7
//     6  3



//     对于具体实现，第二种实现方式，CUBLAS_OP_T，实现矩阵乘法
//     计算矩阵乘法 C = A^T @ B^T
//     定义矩阵A(3x4)
//     0  1  3  3
//     1  0  1  0
//     2  1  2  0

//     由于是列优先，因此写下来如下：
//     A = {0, 1, 2, 1, 0, 1, 3, 1, 2, 3, 0, 0}

//     定义矩阵B(2x3)
//     2  3  0
//     1  2  1
//     B = {2, 1, 3, 2, 0, 1}

//     计算矩阵乘法 C = A^T @ B^T,  C = [4x3] @ [3x2] = [4x2], C.shape = 4x2
//     cublasSgemm(
//         handle,
//         transa = CUBLAS_OP_T, transb = CUBLAS_OP_T, 
//         m = op(A).row = 4,
//         n = op(B).col = 2,
//         k = op(A).col or op(B).row = 3,            
//         alpha,
//         a pointer,
//         lda = A.row = 3,
//         b pointer,
//         ldb = B.row = 2,
//         beta,
//         c pointer,
//         ldc = C.row = 4
//     )
//     计算结果为：3, 2, 9, 6, 4, 2, 7, 3，转换为4x2，列优先为：
//     3  4
//     2  2
//     9  7
//     6  3



//     对于具体实现，第三种实现方式，输入输出的矩阵全部描述为行优先（包括数据以及维度）
//     计算矩阵乘法 C = (AB).T，这里的ABC是行优先描述的
//     定义矩阵A(4x3)  行优先
//     0  1  2
//     1  0  1
//     3  1  2
//     3  0  0
//     A = {0, 1, 2, 1, 0, 1, 3, 1, 2, 3, 0, 0}

//     定义矩阵B(3x2)  行优先
//     2  1
//     3  2
//     0  1
//     B = {2, 1, 3, 2, 0, 1}

//     行优先视角看：计算矩阵乘法 C = (AB).T,  C = [4x3] @ [3x2] = [4x2], C.shape = 4x2
//     实现思路：
//      把输入的A解释为cublas的矩阵描述。这里是行优先转列优先描述
//      cublas_A = A.T = [row = 3, col = 4]

//      按照3行4列解释a pointer中的数据，组织成cublas矩阵
//      cublas_A = {
//          0 1 3 3
//          1 0 1 0 
//          2 1 2 0
//      }

//      把输入的B解释为cublas的矩阵描述。这里是行优先转列优先描述
//      cublas_B = B.T = [row = 2, col = 3]

//      按照3行4列解释a pointer中的数据，组织成cublas矩阵
//      cublas_B = {
//          2 3 0
//          1 2 1
//      }

//      cublas_C = cublas_A.T @ cublas_B.T = [4x3] @ [3x2] = [4x2]
//      cublas_C = {  列优先
//          3,  4 
//          2,  2
//          9,  7
//          6,  3
//      }
//      cublas_C_memory = {3, 2, 9, 6, 4, 2, 7, 3}
     
//      把cublas_C，解释为行优先的输出结果C[2x4]
//      output_C = {
//          3, 2, 9, 6,
//          4, 2, 7, 3
//      }
//      return output_C

//     cublas_A = 解释为cublas(A)
//     cublas_B = 解释为cublas(B)
//     cublas_C = cublas_A.T @ cublas_B.T = [4x3] @ [3x2] = [4x2]

//     cublasSgemm(
//         handle,
//         transa = CUBLAS_OP_T, transb = CUBLAS_OP_T, 
//         m = op(cublas_A).row = 4,
//         n = op(cublas_B).col = 2,
//         k = op(cublas_A).col or op(cublas_B).row = 3,            
//         alpha,
//         a pointer,
//         lda = cublas_A.row = 3,
//         b pointer,
//         ldb = cublas_B.row = 2,
//         beta,
//         c pointer,
//         ldc = cublas_C.row = 4
//     )

//     output_C = 解释为行优先(cublas_C)
//     计算结果为：3, 2, 9, 6, 4, 2, 7, 3，解释为2x4行优先：
//     3, 2, 9, 6,
//     4, 2, 7, 3
//     */
// }

// int main(){

//     int ids = 3000000;
//     int feature_length = 512;
//     int query_count = 1;
//     int arow = ids;
//     int acol = feature_length;
//     int brow = feature_length;
//     int bcol = query_count;
//     int crow = arow;
//     int ccol = bcol;

//     // 准备4x3矩阵A
//     float* a_host = new float[arow * acol];
//     size_t a_bytes = arow * acol * sizeof(float);

//     // 准备3x2矩阵B
//     float* b_host = new float[brow * bcol];
//     size_t b_bytes = brow * bcol * sizeof(float);

//     // 准备4x2矩阵C
//     float* c_host = new float[crow * ccol];
//     size_t c_bytes = crow * ccol * sizeof(float);

//     // 定义device内存，并进行复制
//     float* a_device = nullptr;
//     float* b_device = nullptr;
//     float* c_device = nullptr;
//     cublasHandle_t cublas_handle = nullptr;
//     cudaStream_t stream = nullptr;
//     cudaEvent_t start, stop;
//     cudaStreamCreate(&stream);
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cublasCreate(&cublas_handle);
//     cublasSetStream(cublas_handle, stream);

//     cudaMalloc(&a_device, a_bytes);
//     cudaMalloc(&b_device, b_bytes);
//     cudaMalloc(&c_device, c_bytes);

//     cudaMemcpyAsync(a_device, a_host, a_bytes, cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(b_device, b_host, b_bytes, cudaMemcpyHostToDevice, stream);

//     cudaEventRecord(start, stream);
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

//     cudaEventRecord(stop, stream);
//     cudaMemcpyAsync(c_host, c_device, sizeof(c_host), cudaMemcpyDeviceToHost, stream);
    
//     float ms = 0;
//     cudaEventElapsedTime(&ms, start, stop);
//     printf("耗时:  %.5f ms\n", ms);

//     // for(int p = 0; p < crow * ccol; ++p){
//     //     printf("%f ", c_host[p]);
//     // }
//     // printf("\n");

//     // printf("按照人类习惯打印\n");
//     // for(int row_index = 0; row_index < crow; ++row_index){
//     //     for(int col_index = 0; col_index < ccol; ++col_index){
//     //         // row -> 0, col -> 0
//     //         // p = 0
//     //         // row -> 1, col -> 1
//     //         // crow = 4,   ccol = 2
//     //         // p = row + crow * col
//     //         int p = row_index + crow * col_index;
//     //         printf("%f ", c_host[p]);
//     //     }
//     //     printf("\n");
//     // }
//     return 0;
// }