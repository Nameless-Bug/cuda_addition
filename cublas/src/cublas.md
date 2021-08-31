# cublas<t>gemm()

```C++
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
cublasStatus_t cublasCgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex       *alpha,
                           const cuComplex       *A, int lda,
                           const cuComplex       *B, int ldb,
                           const cuComplex       *beta,
                           cuComplex       *C, int ldc)
cublasStatus_t cublasZgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,
                           cuDoubleComplex *C, int ldc)
cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc)
```

This function performs the matrix-matrix multiplication
- C = α op ( A ) op ( B ) + β C

where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A

op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T A H if  transa == CUBLAS_OP_C

and op ( B ) is defined similarly for matrix B .


<table cellpadding="4" cellspacing="0" summary="" class="table" frame="border" border="1" rules="all">
    <thead class="thead" align="left">
    <tr class="row">
        <th class="entry" valign="top" width="10.460953233385546%" id="d303e28140" rowspan="1" colspan="1">
            Param.
        </th>
        <th class="entry" valign="top" width="18.505258447079882%" id="d303e28143" rowspan="1" colspan="1">
            Memory
        </th>
        <th class="entry" valign="top" width="9.923920340120834%" id="d303e28146" rowspan="1" colspan="1">
            In/out
        </th>
        <th class="entry" valign="top" width="61.109867979413735%" id="d303e28149" rowspan="1" colspan="1">
            Meaning
        </th>
    </tr>
    </thead>
    <tbody class="tbody">
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">handle</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">handle to the cuBLAS library context.</p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">transa</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">operation op(<samp class="ph codeph">A</samp>) that is non- or (conj.) transpose.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">transb</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">operation op(<samp class="ph codeph">B</samp>) that is non- or (conj.) transpose.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">m</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">number of rows of matrix op(<samp class="ph codeph">A</samp>) and <samp class="ph codeph">C</samp>.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">n</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">number of columns of matrix op(<samp class="ph codeph">B</samp>) and <samp class="ph codeph">C</samp>.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">k</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">number of columns of op(<samp class="ph codeph">A</samp>) and rows of op(<samp class="ph codeph">B</samp>).
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">alpha</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">
            <p class="p">host or device</p>
        </td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">&lt;type&gt; scalar used for multiplication.</p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">A</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">
            <p class="p">device</p>
        </td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">&lt;type&gt; array of dimensions <samp class="ph codeph">lda x k</samp> with <samp class="ph codeph">lda&gt;=max(1,m)</samp> if <samp class="ph codeph">transa == CUBLAS_OP_N</samp> and <samp class="ph codeph">lda x m</samp> with <samp class="ph codeph">lda&gt;=max(1,k)</samp> otherwise.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">lda</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">储存矩阵A的前导维数、leading dimension of two-dimensional array used to store the matrix <samp class="ph codeph">A</samp>.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">B</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">
            <p class="p">device</p>
        </td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">&lt;type&gt; array of dimension <samp class="ph codeph">ldb x n</samp> with <samp class="ph codeph">ldb&gt;=max(1,k)</samp> if <samp class="ph codeph">transb == CUBLAS_OP_N</samp> and <samp class="ph codeph">ldb x k</samp> with <samp class="ph codeph">ldb&gt;=max(1,n)</samp> otherwise.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">ldb</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">储存矩阵B的前导维数、leading dimension of two-dimensional array used to store matrix <samp class="ph codeph">B</samp>.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">beta</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">
            <p class="p">host or device</p>
        </td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">&lt;type&gt; scalar used for multiplication. If <samp class="ph codeph">beta==0</samp>, <samp class="ph codeph">C</samp> does not have to be a valid input.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">C</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">
            <p class="p">device</p>
        </td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">in/out</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">&lt;type&gt; array of dimensions <samp class="ph codeph">ldc x n</samp> with <samp class="ph codeph">ldc&gt;=max(1,m)</samp>.
            </p>
        </td>
    </tr>
    <tr class="row">
        <td class="entry" valign="top" width="10.460953233385546%" headers="d303e28140" rowspan="1" colspan="1">
            <p class="p">ldc</p>
        </td>
        <td class="entry" valign="top" width="18.505258447079882%" headers="d303e28143" rowspan="1" colspan="1">&nbsp;</td>
        <td class="entry" valign="top" width="9.923920340120834%" headers="d303e28146" rowspan="1" colspan="1">
            <p class="p">input</p>
        </td>
        <td class="entry" valign="top" width="61.109867979413735%" headers="d303e28149" rowspan="1" colspan="1">
            <p class="p">储存矩阵C的前导维数、leading dimension of a two-dimensional array used to store the matrix <samp class="ph codeph">C</samp>.
            </p>
        </td>
    </tr>
    </tbody>
</table>

# The possible error values returned by this function and their meanings are listed below.
<div class="tablenoborder">
    <table cellpadding="4" cellspacing="0" summary="" class="table" frame="border" border="1" rules="all">
        <thead class="thead" align="left">
        <tr class="row">
            <th class="entry" valign="top" width="50%" id="d303e28594" rowspan="1" colspan="1">
                Error Value
            </th>
            <th class="entry" valign="top" width="50%" id="d303e28597" rowspan="1" colspan="1">
                Meaning
            </th>
        </tr>
        </thead>
        <tbody class="tbody">
        <tr class="row">
            <td class="entry" valign="top" width="50%" headers="d303e28594" rowspan="1" colspan="1">
                <p class="p"><samp class="ph codeph">CUBLAS_STATUS_SUCCESS</samp></p>
            </td>
            <td class="entry" valign="top" width="50%" headers="d303e28597" rowspan="1" colspan="1">
                <p class="p">the operation completed successfully</p>
            </td>
        </tr>
        <tr class="row">
            <td class="entry" valign="top" width="50%" headers="d303e28594" rowspan="1" colspan="1">
                <p class="p"><samp class="ph codeph">CUBLAS_STATUS_NOT_INITIALIZED</samp></p>
            </td>
            <td class="entry" valign="top" width="50%" headers="d303e28597" rowspan="1" colspan="1">
                <p class="p">the library was not initialized</p>
            </td>
        </tr>
        <tr class="row">
            <td class="entry" valign="top" width="50%" headers="d303e28594" rowspan="1" colspan="1">
                <p class="p"><samp class="ph codeph">CUBLAS_STATUS_INVALID_VALUE</samp></p>
            </td>
            <td class="entry" valign="top" width="50%" headers="d303e28597" rowspan="1" colspan="1">
                <p class="p">the parameters <samp class="ph codeph">m,n,k&lt;0</samp></p>
            </td>
        </tr>
        <tr class="row">
            <td class="entry" valign="top" width="50%" headers="d303e28594" rowspan="1" colspan="1">
                <p class="p"><samp class="ph codeph">CUBLAS_STATUS_ARCH_MISMATCH</samp></p>
            </td>
            <td class="entry" valign="top" width="50%" headers="d303e28597" rowspan="1" colspan="1">
                <p class="p">in the case of <samp class="ph codeph">cublasHgemm</samp> the device does not support math in half precision.
                </p>
            </td>
        </tr>
        <tr class="row">
            <td class="entry" valign="top" width="50%" headers="d303e28594" rowspan="1" colspan="1">
                <p class="p"><samp class="ph codeph">CUBLAS_STATUS_EXECUTION_FAILED</samp></p>
            </td>
            <td class="entry" valign="top" width="50%" headers="d303e28597" rowspan="1" colspan="1">
                <p class="p">the function failed to launch on the GPU</p>
            </td>
        </tr>
        </tbody>
    </table>
</div>