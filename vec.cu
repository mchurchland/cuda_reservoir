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
#define THR_PER_BLK 256
/**
cublas does column major order
conda activate cuda13
MATHDX=/home/milesc/para/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03  
nvcc -std=c++17 -dlto  -arch=sm_89 -I "$MATHDX/include"   -I "$MATHDX/external/cutlass/include"   vec.cu -o hi   "$MATHDX/lib/libcusolverdx.a"   -lcublas -lcusolver -lcurand

 */



void print_arr(float * d_c,int n,int m){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", d_c[i*m + j]);
        }
        printf("\n");
    }
}

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void potrf_kernel(DataType* A, typename Solver::status_type* info) {

    extern __shared__ unsigned char shared_mem[]; // switched to unsigned char

    DataType* As = reinterpret_cast<DataType*>(shared_mem);

    constexpr auto lda_smem = Solver::lda;
    constexpr auto lda_gmem = Solver::m_size;

    // Load data from global memory to shared memory
    common::io<Solver>::load_a(A, lda_gmem, As, lda_smem);

    Solver().execute(As, info);

    // Store results back to global memory
    common::io<Solver>::store_a(As, lda_smem, A, lda_gmem);
}
template<int Arch,unsigned int unsigned_n>
float * INV_dx(float * d_A) {
    using namespace cusolverdx;
    using Solver = decltype(Size<unsigned_n, unsigned_n>() + Precision<float>() + Type<type::real>() + Function<potrf>() + LeadingDimension<unsigned_n>() + SM<Arch>() +
                            BlockDim<256>() + FillMode<fill_mode::upper>() + Block());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m == n, "potrf is for Hermitian positive-definite matrix matrix only");
    constexpr auto lda_smem = Solver::lda;

    constexpr auto lda        = m; // this is the leading dimension in global memory for A
    constexpr auto input_size = lda * n; // input global memory size for A

    std::cout << "Use compile-time leading dimension LDA for shared memory = " << lda_smem << std::endl;

    cudaStream_t stream = nullptr;
    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> L(input_size);
    int                    info   = 0;
    int*                   d_info = nullptr; /* error info */

    (cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

    (cudaFuncSetAttribute(potrf_kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));
    //Invokes kernel
    potrf_kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, d_info);

    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    (cudaStreamSynchronize(stream));

    //=========================
    // cuSolver reference
    //=========================
    // Use dumb B as only factorization is performed
    (cudaFree(d_info));
    return d_A;
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
float * matrix_dev_to_cuda(float h_[size_n][size_m],int n){
    size_t bytes = n*n * sizeof(float);
    float  *d_ = nullptr;

    cudaMalloc(&d_, bytes);
    cudaMemcpy(d_, h_, bytes, cudaMemcpyHostToDevice);
    return d_;
}

template < int size_n, int size_m>
void set_arr(float * h_a,int n,int m,float arr[size_n][size_m]){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            h_a[i*m + j]= arr[i][j];
        }
    }
}
template < int size_n, int size_m>
void rand_arr(float arr[size_n][size_m],int n,int m){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(1, 5); // Range [1, 100]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            arr[i][j] = dis(gen);
        }
    }
}
template < int size_n>
void rand_vec(float arr[size_n],int n){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(1, 5); // Range [1, 100]
    for (int i = 0; i < n; i++) {
            arr[i] = dis(gen);
    }
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
float *XTY(float * X, float * Y,int m, int n,int k ){
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes        = m*n * sizeof(float);
    float  *d_c;

    cudaMalloc(&d_c, bytes);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n ,k, &alpha, X, m, Y,k ,&beta, d_c, m);

    cublasDestroy(handle);

    return d_c;
}
float *XY(float * X, float * Y,int m, int n,int k ){
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes        = m*n * sizeof(float);
    float  *d_c;
    cudaMalloc(&d_c, bytes);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k, &alpha, X, m, Y,k ,&beta, d_c, m);
    
    cublasDestroy(handle);

    return d_c;
}


float *XPY(float * X, float * Y,int m, int n){
    // this is assuming x and y are square and the same size
    const float alpha = 1.0f;
    const float beta  = 1.0f;
    size_t bytes        = n*n * sizeof(float);
    float  *d_c;

    cudaMalloc(&d_c, bytes);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, X, n, &beta, Y, n, d_c, n);
    printf("\n\n");
    cublasDestroy(handle);
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




float * diag(float val,int n){ 
    size_t bytes = n*n * sizeof(float);
    float *d_c = nullptr;
    cudaMalloc(&d_c, bytes);
    int blk_in_grid = 256; // this might be problematic
    cuda_diag<<< blk_in_grid, THR_PER_BLK >>>(d_c, n, val); // this is bad , could be easily better cause they all have to do n*n
    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}

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
    float * d_B = diag(1,n); // this is writing over
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
    (cudaFree(d_A));
    (cudaFree(d_Ipiv));
    (cudaFree(d_info));
    (cudaFree(d_work));

    //(cusolverDnDestroy(cusolverH));

    //(cudaStreamDestroy(stream));

    return d_B;
}


template <unsigned int size_n, unsigned int size_m>
float * ridge_reg(float  X [size_n][size_m], float * Y,int n,int m,float alpha){
    //assuming that Y is a vector maybe I want to add a case for this
    size_t bytes        = n*m*2 * sizeof(float);
    size_t vec_bytes        = n * sizeof(float);
    
    
    float *h_a = nullptr;
    float *h_b = nullptr;
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(vec_bytes);


    set_arr<size_n,size_m>(h_a,n,m, X);
    set_vec(h_b,n, Y);
    


    float *d_a = nullptr;
    float *d_b = nullptr;
    cudaMalloc(&d_a, bytes); //these are the cuda arrays
    cudaMalloc(&d_b, vec_bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice);

    //allocate memory needed for the computation
    // I feel like this could be bad these are alocated on host memory but they are 
    //just storing a pointer to device memory maybe they can be maller in size, like just the size of a pointer to a float
    float  *h_xtx = nullptr;
    float  *h_xtxpLAMID = nullptr;
    float  *h_xtx_p_LAMID_INV = nullptr;
    float  *h_xty = nullptr;
    float  * h_weights = (float *)malloc(vec_bytes);
    


    std::future<float *> para_xtx = std::async(std::launch::async, XTY, d_a,d_a,n,n,m);  //do these in parallel
    std::future<float *> para_id = std::async(std::launch::async, diag, alpha,n); 
    std::future<float *> para_xty = std::async(std::launch::async, XTY, d_a,d_b,m,n,1); 

    float * id = para_id.get(); // these are all kept on device
    h_xtx = para_xtx.get(); 
    printf("fdsaz");
    print_cuda(h_xtx,n,n);
    printf("\n");
    h_xtxpLAMID = XPY(h_xtx,id,n,n); 

    print_cuda(h_xtxpLAMID,n,n);
        printf("\n");

    h_xtx_p_LAMID_INV = XINV(h_xtxpLAMID,n);
    print_cuda(h_xtx_p_LAMID_INV,n,m);
    printf("fdsaz");
    //h_xtx_p_LAMID_INV  = INV_dx<890,size_n>(h_xtxpLAMID);
    
    h_xty = para_xty.get();
    h_weights = XY(h_xtx_p_LAMID_INV,h_xty,m,n,1); //thios is a vector // \addtogrou
    

    cudaFree(h_xtx);

    //cudaFree(h_xtxpLAMID);

    cudaFree(h_xtx_p_LAMID_INV);

    cudaFree(h_xty);

    cudaFree(id);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);


    return h_weights;
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
        i += gridSize; //somehow this is treating them like integers
        //printf("tid %d, val %f i %d, i+block %d \n ", tid, sdata[tid],i,i+blockSize); // this dosent when when i=i+blocksize
   
    }

    
    // talk to jim fix about this I want to understand
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid); // do it until we are small enough to fit in a warp
    if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
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
    elementwise<<< blk_in_grid, THR_PER_BLK >>>(Y, X,d_out,n);
    return d_out;
}

float sum(float * Y,int n,int blk_in_grid){ // y needs to be a cuda vector
    size_t vec_bytes        = n * sizeof(float);
    float *d_out = nullptr;
    cudaMalloc(&d_out, vec_bytes);

    float * arr = vec_cuda_to_dev(Y,n);

    float sum = std::accumulate(arr, arr + n, 0.0f);// can be removed after testing
    reduce6<THR_PER_BLK><<< blk_in_grid, THR_PER_BLK >>>(Y, d_out,n); // fucked up on floats
    float * h_out =(float *)malloc(vec_bytes);
    cudaMemcpy(h_out, d_out, vec_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    float out = h_out[0];
    free(h_out);
    assert(std::abs(sum-out) < 0.001);
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
    sub_val<<< blk_in_grid, THR_PER_BLK >>>(d_a,d_mean,d_c, n);
    }
    else if (sq==true){
    sub_val_sq<<< blk_in_grid, THR_PER_BLK >>>(d_a,d_mean,d_c, n);
    }
    //reduce6<THR_PER_BLK><<< blk_in_grid, THR_PER_BLK >>>(d_c, d_out,n);
    cudaFree(d_a);
    cudaFree(d_mean);
    free(h_a);
    return d_c;
}

float get_top(float * Y, float * Y_hat, int n,float alpha,int blk_in_grid,size_t vec_bytes){
    return sum(mult(Y,Y_hat,n,blk_in_grid),n,blk_in_grid);
}

float get_bot(float * Y, float * Y_hat, int n,float alpha,int blk_in_grid,size_t vec_bytes){

    return std::sqrt(sum(Y,n,blk_in_grid) * sum(Y_hat,n,blk_in_grid)) + alpha; // bad i hate i hate u i hate u bot is wrong
}
float r2_score(float * Y, float * Y_hat, int n,float alpha){
    size_t vec_bytes        = n * sizeof(float);
	int blk_in_grid = ceil( float(n) / THR_PER_BLK );
    //need to accutally free memory here im being a bad bad boy

    float * yt = (float *)malloc(vec_bytes);
    float * yh = (float *)malloc(vec_bytes);
    float * yt_sq = (float *)malloc(vec_bytes);
    float * yh_sq = (float *)malloc(vec_bytes);
    
    std::future<float *> Y_tru = std::async(std::launch::async, dif, Y,n,blk_in_grid,false); 
    yt = Y_tru.get();
    std::future<float *> Y_ha = std::async(std::launch::async, dif, Y_hat,n,blk_in_grid,false); 
     yh = Y_ha.get();
    std::future<float *> Y_tru_sq = std::async(std::launch::async, dif, Y,n,blk_in_grid,true); 
    yt_sq = Y_tru_sq.get();
    std::future<float *> Y_ha_sq = std::async(std::launch::async, dif, Y_hat,n,blk_in_grid,true); 
    yh_sq = Y_ha_sq.get();
    //yt = Y_tru.get();
    //yh = Y_ha.get();
    //yt_sq = Y_tru_sq.get();
    //yh_sq = Y_ha_sq.get();


    std::future<float> para_top = std::async(std::launch::async, get_top, yt,yh,n,alpha,blk_in_grid,vec_bytes); 
    std::future<float> para_bot = std::async(std::launch::async, get_bot, yt_sq,yh_sq,n,alpha,blk_in_grid,vec_bytes); 
    float top = para_top.get();
    float bot = para_bot.get();
    printf("top %f, bot %f\n", top,bot);
    return std::pow((top/bot), 2);
}




int main(void)
{
    
    // Error code to check return values for CUDA calls
    const int n = 1 <<2; // dosent work for not nice cases
    const int m = 1 <<1;
    constexpr int N =  n;
    constexpr int M =  m;

    float matrix_A [n][m] ={};
    float matrix_B [n]= {};
    rand_arr<N,M>(matrix_A,n,m);
    rand_vec<N>(matrix_B,n);
    
    
    //make seperate ones for train and test 
    //we are going to chop different tau amounts this is probably done on cpu in parallel but the ridge and r2 on the cpu
    // will need to be careful because we will stop working with square matrices
    // we dont actually *NEED* to slice the input matrix, we just need to tell our function that our matrix is smaller than it is !!!!
    //        yte = ute[:-tau]
    //        Xtr_d = Xtr[tau:] ##so that they are the same dimensiobn


    float  *h_weights = nullptr;    
    auto start = std::chrono::steady_clock::now();
    h_weights = ridge_reg<N,M>(matrix_A,matrix_B,n,m,0.0001);
    print_cuda_vec(h_weights,N);
    printf("\n\n\n");
    float * mat = XY(matrix_dev_to_cuda<N,M>(matrix_A,n),h_weights,n,m,1); 
    float * matrix_C = vec_cuda_to_dev(mat,n); 
    // this is getting our y hat
    //this gives me different answers at times, idk why 
    print_vec(matrix_C,n); // this is fucked  this should be
    printf("\n\n\n");
    printf("r2 score %f\n",  r2_score(matrix_B,matrix_C,n,0.0001));
    //print_vec(matrix_C,n);

    auto end = std::chrono::steady_clock::now();

    // 3. Calculate the difference (duration)
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf( "Time elapsed: %lf", elapsed.count() );
    cudaDeviceReset();
    printf("Done\n");
    return 0;
}
