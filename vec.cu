#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <curand.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cmath> // Required for sqrt
#include <future>
#include <cusolverDn.h>
#include <cusolverdx.hpp>
#include "/home/milesc/para/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03/example/cusolverdx/common/device_io.hpp"
#include <vector>
#include <typeinfo>
#include <chrono>
#define THR_PER_BLK 128
#define BLK_IN_GRID 64
/**
cublas does column major order
conda activate cuda13
MATHDX=/home/milesc/para/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03  
nvcc -std=c++17 -dlto  -arch=sm_89 -I "$MATHDX/include"   -I "$MATHDX/external/cutlass/include"   vec.cu -o hi   "$MATHDX/lib/libcusolverdx.a"   -lcublas -lcusolver -lcurand

 */

void print_arr(float * d_c,int n,int m){
    //https://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", d_c[j*n + i]);
            }
        }
        printf("\n");
    }




void print_vec(float * d_c,int n){
    for (int i = 0; i < n; i++) {
            printf("%f ", d_c[i]);
        }
        printf("\n");
    }

void print_cuda(float * h_c,int n,int m){
    size_t bytes = n*m * sizeof(float);
    float * out =(float *)malloc(bytes);
    cudaMemcpy(out, h_c, bytes, cudaMemcpyDeviceToHost);
    print_arr(out,n,m);
}

void print_cuda_vec(float * d_c,int n){
    size_t bytes = n * sizeof(float);
    
    float * out =(float *)malloc(bytes);
    cudaMemcpy(out, d_c, bytes, cudaMemcpyDeviceToHost);
    print_vec(out,n);
}

float * vec_cuda_to_dev(float * d_,int n){
    size_t bytes = n * sizeof(float);
    float * out =(float *)malloc(bytes);
    cudaMemcpy(out, d_, bytes, cudaMemcpyDeviceToHost);
    return out;
}

float * vec_dev_to_cuda(float * h_,int n){
    size_t bytes = n * sizeof(float);
    float  *d_ = nullptr;

    cudaMalloc(&d_, bytes);
    cudaMemcpy(d_, h_, bytes, cudaMemcpyHostToDevice);
    return d_;
}


template < int size_n, int size_m>
void set_arr(float * h_a,int n,int m,float arr[size_n][size_m]){
    //https://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
    for (int i = 0; i < n; i++) { // this is the row
        for (int j = 0; j < m; j++) { // this is the col
            h_a[j*n + i]= arr[i][j];
        }
    }
}
template < int size_n, int size_m>
float * matrix_host_to_cuda(float h_[size_n][size_m],int n,int m){
    size_t bytes = n*m * sizeof(float);
    float  *d_ = nullptr;
    float * h = (float *)malloc(bytes);
    
    set_arr<size_n,size_m>(h,n,m, h_);
    
    cudaMalloc(&d_, bytes);
    cudaMemcpy(d_, h, bytes, cudaMemcpyHostToDevice);
    return d_;
}
template < int size_n, int size_m>
void rand_arr(float arr[size_n][size_m],int n,int m){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(-100, 100); // Range [1, 100] doing 1,100 is problematic and leads to solves not working
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            arr[i][j] = dis(gen);
            //printf("%f ",arr[i][j]);
        }
        //printf("\n");
    }
}
template < int size_n>
void rand_vec(float arr[size_n],int n){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(-100, 100); // Range [1, 100] doing 1,100 is problematic and leads to solves not working, google’s built-in AI summary helped me here 
    // with a search about
    // the probability of a 1,100 sampled matrix bieng degenerate
    // it helped me realize that the problem is not the matrix occasionally not having full rank but the **amplification** of floating point numbers
    // with 1,100 the floats get super large (about 2/100 times) in xtx and thus lead to problems
    // it turns out that Floats are generally only reliable to 6-7 significant decimal digits. and thus here occasionally break when the numbers in the matrix are large
    // having an distribution centered at 0 changes this as the negative numbers "dampen" the amplification during xtx
    // 
    for (int i = 0; i < n; i++) {
            arr[i] = dis(gen);
            //printf("%f ",arr[i]);
    }
    //printf("\n");
}
void set_vec(float * h_a,int n,float arr[]){
    for (int i = 0; i < n; i++) {
            h_a[i]= arr[i];
    }
}

float * get_vec(float * h_a,int n){
    float *arr = nullptr;
    arr = (float *)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) {
            arr[i] = h_a[i];
    }
    return arr;
}
void set_vec_val(float * h_a,int n,float val){
    for (int i = 0; i < n; i++) {
            h_a[i]= val;
    }
}
float *XTY_s(cublasHandle_t handle, cudaStream_t stream,
             float *X, float *Y, int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes = m * n * sizeof(float);
    float *d_c = nullptr;
    cudaMalloc(&d_c, bytes);

    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k, &alpha, X, k, Y, k, &beta, d_c, m);
    return d_c;
}

template<unsigned M, unsigned N, unsigned NT>
__global__ __launch_bounds__(NT) void inv_kernel(float* A, unsigned lda, float* B, float* x, int* info) {

    // To maximize parallelism, we compute [G v] = A' * [A b] with a single GEMM, then solve Gx = v
    // The matrices are assumed to be stored in column-major format
    // (M + N) * (N + 1) words of dynamic shared memory are required Ax = ID, x is the inverse
    // Setup shared memory
    extern __shared__ __align__(16) cusolverdx::byte shared_mem[]; // not sure why im doing 16: its for 128 bit
    // Slice shared memory into pointers
    // Note that in memory, we have the augmented matrices [As Bs]
    auto [As, Bs] = cusolverdx::shared_memory::slice<float, float>(
        shared_mem,
        alignof(float), M * M,
        alignof(float), M*M
    );
    __syncthreads();

    using          prec = float;

    using POSV = decltype(cusolverdx::Function<cusolverdx::posv>() + cusolverdx::Size<N, N, N>() + cusolverdx::FillMode<cusolverdx::lower>() +
                          cusolverdx::Precision<prec>() + cusolverdx::Type<cusolverdx::type::real>() + cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<890>());

    //// Load data from global memory
    __syncthreads();
    cusolverdx::copy_2d<NT, M, M, cusolverdx::arrangement::col_major>(A, lda, As, M);// memory loads are good
    cusolverdx::copy_2d<NT, M, M, cusolverdx::arrangement::col_major>(B, lda, Bs, M);


    __syncthreads();
    POSV().execute(As, Bs, info);

    //// store solution back to global memory
    __syncthreads();
    cusolverdx::copy_2d<NT, N, N, cusolverdx::arrangement::col_major>(Bs, N, x, N); // 1d solve
    //if (*info !=0){printf("%d %f\n", * info,A[4095]);
   // }
    
}
template<int Arch,unsigned int unsigned_n>
float * INV_dx(float * d_A,float * d_B) {
    
    using namespace cusolverdx;
    using Solver = decltype(Size<unsigned_n, unsigned_n>() + Precision<float>() + Type<type::real>() + Function<posv>() + LeadingDimension<unsigned_n>() + SM<Arch>() +
                            BlockDim<BLK_IN_GRID>() + FillMode<fill_mode::lower>() + Block());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m == n, "posv is for Hermitian positive-definite matrix matrix only");
    constexpr auto lda_smem = Solver::lda;
    float * out = nullptr;
    constexpr auto lda        = m; // this is the leading dimension in global memory for A
    constexpr auto input_size = lda * n; // input global memory size for A

    //std::cout << "Use compile-time leading dimension LDA for shared memory = " << lda_smem << std::endl;


    std::vector<data_type> L(input_size);
    int                    info   = 0;
    int*                   d_info = nullptr; /* error info */

    (cudaMalloc(&d_info, sizeof(int)));
    cudaMalloc(&out, n*m*sizeof(float));
    constexpr unsigned ne_required_smem = ((m + n) * (n + 1) + n) * sizeof(data_type); // this might be able to be adjusted

        size_t bytes = n*m * sizeof(float);

    //Invokes kernel
    //constexpr unsigned ne_NT   = 32;
    using KernelPtr = void (*) (float *, unsigned, float *, float *, int *);
    KernelPtr ne_kernel = inv_kernel<unsigned_n,unsigned_n,THR_PER_BLK>; // this needs to be 32 for some reason otherwise it fails at times :(
    (cudaFuncSetAttribute(ne_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ne_required_smem));
    //block in grid thread in block
    //constexpr unsigned batches = 1;
    cudaStream_t stream = nullptr;
    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaGetLastError();
    const auto run_ne_d_impl = [&](cudaStream_t str) {
        ne_kernel<<<1, THR_PER_BLK, ne_required_smem, str>>>(d_A,unsigned_n,d_B,out,d_info);
    };
    
    run_ne_d_impl(stream);

    //print_cuda(out,m,n);
    //printf("dead?");
        //cudaError_t err = cudaGetLastError();
        
        #define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        fprintf(stderr,                                           \
            "CUDA error at %s:%d: %s (%s)\n",                     \
            __FILE__, __LINE__,                                   \
            cudaGetErrorName(err__), cudaGetErrorString(err__));   \
        exit(1);                                                   \
    }                                                             \
} while (0)
CUDA_CHECK(cudaPeekAtLastError());     // catches launch/config argument errors
CUDA_CHECK(cudaDeviceSynchronize());   // catches runtime/device-side errors


    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    (cudaStreamSynchronize(stream));

    //=========================
    // cuSolver reference
    //=========================
    // Use dumb B as only factorization is performed
    cudaStreamDestroy(stream);
    (cudaFree(d_info));
    return out;
}

float *XY_s(cublasHandle_t handle, cudaStream_t stream,
            float *X, float *Y, int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes = m * n * sizeof(float);
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, &alpha, X, m, Y, k, &beta, d_c, m);
    return d_c;
}

float *XPY_s(cublasHandle_t handle, cudaStream_t stream,
             float *X, float *Y, int m, int n)
{
    const float alpha = 1.0f;
    const float beta  = 1.0f;
    size_t bytes = m * n * sizeof(float);
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cublasSetStream(handle, stream);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, &alpha, X, m, &beta, Y, m, d_c, m);
    return d_c;
}

__global__ void cuda_diag(float * c, int n,float val)
{    
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id <= n*n){
	if((id+1)%(n+1) ==1){
        c[id] = val;
        
        }
    else{
        c[id] =0;
    }}
}

float *diag_s(cudaStream_t stream, float val, int n)
{
    size_t bytes = n * n * sizeof(float);
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    int total_elements = n * n;
    int blocks = (total_elements + THR_PER_BLK - 1) / THR_PER_BLK;
    cuda_diag<<<blocks, THR_PER_BLK, 0, stream>>>(d_c, n, val);
    return d_c;
}
float * XINV(float * d_A, int n){ // somehow these get really big sometimes this dosent work but only sometimes :)

    cudaStream_t stream = NULL;
    const int lda = n;
    const int ldb = n;
    std::vector<float> LU(lda * n, 0);
    std::vector<int> Ipiv(n, 0);
    size_t bytes = n*n * sizeof(float);
    int info = 0;
    //float * d_B = diag_s(1,n); // this is writing over
    //float *d_B = nullptr; /* device copy of B */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    float *d_work = nullptr; /* device workspace for getrf */
    const int pivot_on = 1;
    /* step 1: create cusolver handle, bind a stream */
    cusolverDnHandle_t cusolverH = NULL;
    
    (cusolverDnCreate(&cusolverH));
    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    (cusolverDnSetStream(cusolverH, stream));
    float * d_B = diag_s(stream,1,n); // this is writing over

    
    (cudaMalloc(&d_Ipiv, sizeof(int) * Ipiv.size()));
    (cudaMalloc(&d_info, sizeof(int)));
    /* step 3: query working space of getrf */
    (cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &lwork));

    (cudaMalloc(&d_work, sizeof(float) * lwork));
    /* step 4: LU factorization */
    if (pivot_on) {
        (cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, d_Ipiv, d_info));
    } else {
        (cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, NULL, d_info));
    }
    if (pivot_on) {
        (cudaMemcpy(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost));
    }
    (
        cudaMemcpyAsync(LU.data(), d_A, bytes, cudaMemcpyDeviceToHost, stream));

    (cudaStreamSynchronize(stream));
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    if (pivot_on) {
        //signature
        (cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, /* nrhs */
                                        d_A, lda, d_Ipiv, d_B, ldb, d_info));
    } else {
        (cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, /* nrhs */
                                        d_A, lda, NULL, d_B, ldb, d_info));
    }

    (cudaStreamSynchronize(stream));
    (cudaFree(d_Ipiv));
    (cudaFree(d_info));
    (cudaFree(d_work));

    //(cusolverDnDestroy(cusolverH));

    //(cudaStreamDestroy(stream));

    return d_B;
}


template <unsigned int size_n, unsigned int size_m>
float *ridge_reg(float X[size_n][size_m], float *Y, int n, int m, float alpha,
                 cublasHandle_t handle, cudaStream_t *streams)
{
    size_t bytes     = n * m * sizeof(float);
    size_t vec_bytes = n * sizeof(float);

    // --- Convert host data to column-major and upload to device ---
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(vec_bytes);
    set_arr<size_n, size_m>(h_a, n, m, X);
    set_vec(h_b, n, Y);

    float *d_a = nullptr;
    float *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice));

    // Done with host staging buffers
    free(h_a);
    free(h_b);

    // Unpack streams for readability
    cudaStream_t stream1 = streams[0];
    cudaStream_t stream2 = streams[1];
    cudaStream_t stream3 = streams[2];

    // Stream 1: XTX = X^T * X   (m x m)
    float *d_xtx = XTY_s(handle, stream1, d_a, d_a, m, m, n);

    // Stream 2: diag_id = alpha * I   (m x m)
    float *d_diag_id = diag_s(stream2, alpha, m);

    // Stream 3: XTY = X^T * Y   (m x 1)
    float *d_xty = XTY_s(handle, stream3, d_a, d_b, m, 1, n);

    // --- Synchronize: s1 and s2
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // ---  XPY -> INV -> final matmul ---

    // Step: (X^T X) + alpha*I
    float *d_xtx_plus_id = XPY_s(handle, stream1, d_xtx, d_diag_id, m, m);
    CUDA_CHECK(cudaStreamSynchronize(stream1));  // XPY need to be done, finito before INV reads it

    // Step: Invert (X^T X + alpha*I)
    // Create identity matrix as RHS for the solve
    float *d_inv_rhs = diag_s(stream1, 1.0f, m);
    CUDA_CHECK(cudaStreamSynchronize(stream1));  // diag must finish before INV

    float *d_inv = INV_dx<890, size_m>(d_xtx_plus_id, d_inv_rhs);
    // INV_dx internally sync

    // Step: Wait for XTY (stream3) before dinal math
    CUDA_CHECK(cudaStreamSynchronize(stream3));

    float *d_weights = XY_s(handle, stream1, d_inv, d_xty, m, 1, m);
    CUDA_CHECK(cudaStreamSynchronize(stream1));  // ensure weights are ready

    // --- Trashman cleanup shot
    cudaFree(d_xtx);
    cudaFree(d_diag_id);
    cudaFree(d_xty);
    cudaFree(d_xtx_plus_id);
    cudaFree(d_inv_rhs);
    cudaFree(d_inv);
    cudaFree(d_a);
    cudaFree(d_b);

    // steam cleanup in main they are owned by main()

    return d_weights;  // caller owns this device pointer
}



template <int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    // theres a better way to do the warp reduce
    // for int i=1; i<warpSize; i*=2
    //  value += __shfl_xor_sync(-1,value,i)
    // this makes a butterfly network
    // this leads to all output nodes having the sum
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <int blockSize> //code from nvidia presentation
__global__ void reduce6(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    

    //sometimes g_idata[i+blockSize] is loaded with a copy of the data but only on later times
    while (i < n) {
        //printf("i %d, tid %d, val %f, inval %f inval_text %f\n", i,tid,sdata[tid],g_idata[i],g_idata[i+blockSize]);
        //g_idata why does thise sometimes have stuff and sometimes not
        // need to check thta i + block size is actually in the allocated memory cause its probably not
        sdata[tid] += g_idata[i]; //+ g_idata[i+blockSize]; 
        i += gridSize; // I think this is wrong
        //printf("tid %d, val %f i %d, i+block %d \n ", tid, sdata[tid],i,i+blockSize); // this dosent when when i=i+blocksize
    }

    
    // talk to jim fix about this I want to understand
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid); // do it until we are small enough to fit in a warp
    if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0]; 
}
}


/**
 * Host main routine
 */

__global__ void sub_val(float *a, float *b, float *c, int n)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n){
        c[id] = a[id] - b[0];
        }
}


__global__ void sub_val_sq(float *a, float *b, float *c, int n)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n){
        float sq = (a[id] - b[0]);
        c[id] = sq*sq;
        }
}

__global__ void elementwise(float *a, float *b, float *c, int n)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n){
        c[id] = (a[id] * b[id]);
        }
}


float * mult(float *Y,float *X,int n,int blk_in_grid){
    size_t vec_bytes        = n * sizeof(float);
    float *d_out = nullptr;
    cudaMalloc(&d_out, vec_bytes);
    elementwise<<< BLK_IN_GRID, THR_PER_BLK >>>(Y, X,d_out,n);
    return d_out;
}

float sum(float * Y,int n,int blk_in_grid){ // y needs to be a cuda vector this value can differ from the real sum due to floating point calculations, write about this
    //https://stackoverflow.com/questions/15139656/precision-in-sum-reduction-kernel-with-floats?utm_source=chatgpt.com
    size_t vec_bytes        = n * sizeof(float);
    float *d_out = nullptr;
    cudaMalloc(&d_out, vec_bytes);

    //float * arr = vec_cuda_to_dev(Y,n);

    //float sum = std::accumulate(arr, arr + n, 0.0f);// can be removed after testing
    reduce6<THR_PER_BLK><<< BLK_IN_GRID, THR_PER_BLK >>>(Y, d_out,n); // messed up on floats
    float * h_out =(float *)malloc(vec_bytes);
    cudaMemcpy(h_out, d_out, vec_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    float out = h_out[0];
    free(h_out);
    //printf("out %f\n sum %f\n", out,sum);
    //assert(std::abs(sum-out) < 2);
    return out;
}



float * dif(float * Y, int n,  int blk_in_grid,bool sq)
{
    size_t vec_bytes        = n * sizeof(float);
    

    
    float *h_a = nullptr;
    h_a = (float *)malloc(vec_bytes);


    set_vec(h_a,n, Y);

    float *d_a = nullptr;
    float *d_mean = nullptr;
    float *d_out = nullptr;
    float *d_c = nullptr;
    cudaMalloc(&d_a, vec_bytes); 
    cudaMalloc(&d_c, vec_bytes); 

    cudaMalloc(&d_out, vec_bytes);
    cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice);
    float  mean = sum(d_a,n,blk_in_grid)/n;
    
    cudaMalloc(&d_mean, sizeof(float));

    cudaMemcpy(d_mean, &mean, sizeof(float), cudaMemcpyHostToDevice);
    //print_cuda_vec(d_a,n);
    if (sq==false){
    sub_val<<< BLK_IN_GRID, THR_PER_BLK >>>(d_a,d_mean,d_c, n);
    }
    else if (sq==true){
    sub_val_sq<<< BLK_IN_GRID, THR_PER_BLK >>>(d_a,d_mean,d_c, n);
    }
    //reduce6<THR_PER_BLK><<< blk_in_grid, THR_PER_BLK >>>(d_c, d_out,n);
    cudaFree(d_a);
    cudaFree(d_mean);
    free(h_a);
    return d_c;
}

float get_top(float * Y, float * Y_hat, int n,float alpha,int blk_in_grid,size_t vec_bytes){
    auto a= sum(mult(Y,Y_hat,n,blk_in_grid),n,blk_in_grid);
    return a;
}

float get_bot(float * Y, float * Y_hat, int n,float alpha,int blk_in_grid,size_t vec_bytes){

    return std::sqrt(sum(Y,n,blk_in_grid) * sum(Y_hat,n,blk_in_grid)) + alpha; // bad i hate i hate u i hate u bot is wrong
}
float r2_score(float * Y, float * Y_hat, int n,float alpha){
    size_t vec_bytes        = n * sizeof(float);
	int blk_in_grid = ceil( float(n) / THR_PER_BLK );
    //need to accutally free memory here im being a bad bad boy
    float * yt = nullptr;
    float * yh = nullptr;
    float * yt_sq = nullptr;
    float * yh_sq = nullptr;
    
    //these could be done in parallel with careful gpu shared memory allocation, sum breaks if they are done in parallel rn
    std::future<float *> Y_tru = std::async(std::launch::async, dif, Y,n,blk_in_grid,false); 
        yt = Y_tru.get();
    std::future<float *> Y_ha = std::async(std::launch::async, dif, Y_hat,n,blk_in_grid,false); 
        yh = Y_ha.get();
    std::future<float *> Y_tru_sq = std::async(std::launch::async, dif, Y,n,blk_in_grid,true); 
        yt_sq = Y_tru_sq.get();
    std::future<float *> Y_ha_sq = std::async(std::launch::async, dif, Y_hat,n,blk_in_grid,true); 
    yh_sq = Y_ha_sq.get();
    //doing these sequentially is important because im using shared memory
    //printf("yt %f\n", yt);
    //printf("yh %f\n", yh);
    //printf("yt_sq %f\n", yt_sq);
    //printf("yh_sq %f\n", yh_sq);


    //these could be done in parallel with careful gpu shared memory allocation, sum breaks if they are done in parallel rn
    std::future<float> para_top = std::async(std::launch::async, get_top, yt,yh,n,alpha,blk_in_grid,vec_bytes); 
     float top = para_top.get();
    std::future<float> para_bot = std::async(std::launch::async, get_bot, yt_sq,yh_sq,n,alpha,blk_in_grid,vec_bytes); 
   
    float bot = para_bot.get();
    //printf("top %f, bot %f\n", top,bot);
    return std::pow((top/bot), 2);
}



void test_xy(){
    const int n=2;
    const int m=2;
    const int l=1;
    constexpr int N =  n;
    constexpr int M =  m;
    constexpr int L =  l;
    float matrix_A [2][2]= {{1,2},{3,4}};
    float matrix_B [2][1] ={{1.2},{1.77}};
    //print_cuda(matrix_host_to_cuda<M,N>(matrix_A,m,n),m,n);
    //print_cuda(matrix_host_to_cuda<M,L>(matrix_B,m,l),m,l);
    //float * mat = XY(matrix_host_to_cuda<M,N>(matrix_A,m,n),matrix_host_to_cuda<M,L>(matrix_B,m,l),2,1,2);
    //print_cuda_vec(mat,m);
}




int main(void)
{
    const int n = 1 << 6;  // 64
    const int m = 1 << 6;  // 64

    constexpr int N = n;
    constexpr int M = m;

    float matrix_A[n][m] = {};
    float matrix_B[n] = {};

    
    // Create cuBLAS handle ONCE 
    cublasHandle_t handle;
    cublasCreate(&handle);

    
    // Create 3 CUDA streams 
    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

   
    for (int j = 0; j < 20; j++) {
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < 100; i++) {
            rand_arr<N, M>(matrix_A, n, m);
            rand_vec<N>(matrix_B, n);

            float *d_weights = ridge_reg<N, M>(matrix_A, matrix_B, n, m, 0.0001f,
                                                handle, streams);

            float *d_X = matrix_host_to_cuda<N, M>(matrix_A, n, m);
            float *d_yhat = XY_s(handle, streams[0], d_X, d_weights, n, 1, m);
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));

            // Copy y_hat to host for r2_score
            float *matrix_C = vec_cuda_to_dev(d_yhat, n);

            float r2 = r2_score(matrix_B, matrix_C, n, 0.0001f);
            if (r2 < 0.2) {
                printf("F ");
            }

            // Free up stuff 
            cudaFree(d_weights);
            cudaFree(d_X);
            cudaFree(d_yhat);
            free(matrix_C);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("%lf\n", elapsed.count());
    }

    for (int i = 0; i < 3; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cublasDestroy(handle);

    cudaDeviceReset();
    printf("Done\n");
    return 0;
}