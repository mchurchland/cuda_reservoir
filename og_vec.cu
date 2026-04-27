/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

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
#include <cusolverDn.h>
#include <vector>
#include <typeinfo>
#include <chrono>
#define THR_PER_BLK 256
/**
cublas does column major order  nvcc vec.cu -o hi -lcublas -lcusolver -lcurand
 */
void print_arr(float * d_c,int n){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", d_c[i*n + j]);
        }
        printf("\n");
    }
}

void print_vec(float * d_c,int n){
    for (int i = 0; i < n; i++) {
            printf("%f ", d_c[i]);
        }
        printf("\n");
    }

void print_cuda(float * h_c,int n){
    size_t bytes = n*n * sizeof(float);
    float * out =(float *)malloc(bytes);
    cudaMemcpy(out, h_c, bytes, cudaMemcpyDeviceToHost);
    print_arr(out,n);
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
            h_a[i*n + j]= arr[i][j];
        }
    }
}
template < int size_n, int size_m>
void rand_arr(float arr[size_n][size_m],int n,int m){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(1, 100); // Range [1, 100]
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
    std::uniform_int_distribution<> dis(1, 100); // Range [1, 100]
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
    size_t bytes        = m*k * sizeof(float);
    float  *d_c;

    cudaMalloc(&d_c, bytes);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k ,m, &alpha, X, n, Y,n ,&beta, d_c, n);

    cublasDestroy(handle);

    return d_c;
}
float *XY(float * X, float * Y,int m, int n,int k ){
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes        = n*n * sizeof(float);
    float  *d_c;
    cudaMalloc(&d_c, bytes);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k ,m, &alpha, X, n, Y,n ,&beta, d_c, n);
    
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
    cublasDestroy(handle);

    return d_c;
}


float * diag(float a,int size){ 
    float *h_a =nullptr;
    size_t bytes = size*size * sizeof(float);
    h_a = (float *)calloc(size*size,sizeof(float));
     // for some reason I have to do this and if I dont I get a really stupid error that I cannot figure out save 
    //me jim
    /*
    printf("%d %f %f  %f  %f\n",size,a,h_a[7],h_a[6],h_a[5]);
    print_arr(h_a,size);
    printf("\n");
    */
    for (int i = 0; i < size; i++) { 
            h_a[(i*size)+i]=a;
        }

    float * d_a = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    free(h_a);
    return d_a;
}

float * XINV(float * d_A, int n){
    
    cudaStream_t stream = NULL;

    const int lda = n;
    const int ldb = n;
    std::vector<float> LU(lda * n, 0);
    std::vector<int> Ipiv(n, 0);
    size_t bytes = n*n * sizeof(float);
    int info = 0;
    float * d_B = diag(1,n);
    
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


    /* step 2: copy A to device */
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
        (cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost, stream));
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


template < int size_n, int size_m>
float * ridge_reg(float  X [size_n][size_m], float * Y,int n,int m,float alpha){
    //assuming that Y is a vector maybe I want to add a case for this
    size_t bytes        = n*n * sizeof(float);
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
    


    h_xtx = XTY(d_a,d_a,n,n,n); 
    float * id = diag(alpha,n);
    h_xtxpLAMID = XPY(h_xtx,id,n,n);
    h_xtx_p_LAMID_INV = XINV(h_xtxpLAMID,n);
    h_xty = XTY(d_a,d_b,n,n,1); // fucked up  1,5
    h_weights = XY(h_xtx_p_LAMID_INV,h_xty,n,n,1); //i think thios is a vector


    cudaFree(h_xtx);
    cudaFree(h_xtxpLAMID);
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
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
/**
 * Host main routine
 */
__global__ void sub_vectors(float *a, float *b, float *c, int n)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n){
        float res = a[id] - b[id];
        c[id] = res*res;
        }
}


float get_vec_mean(float * X,int n,  int blk_in_grid){
    size_t vec_bytes        = n * sizeof(float);
    float *d_out = nullptr;
    cudaMalloc(&d_out, vec_bytes);

    float *d_X = nullptr;
    cudaMalloc(&d_X, vec_bytes); 
    cudaMemcpy(d_X, X, vec_bytes, cudaMemcpyHostToDevice);

    float * h_out =(float *)malloc(vec_bytes);
    reduce6<THR_PER_BLK><<< blk_in_grid, THR_PER_BLK >>>(d_X, d_out,n);
    cudaMemcpy(h_out, d_out, vec_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_out);
    float out = h_out[0];
    free(h_out);
    return out/n;
}

float dif_squared(float * Y,float * Y_hat, int n,  int blk_in_grid)
{
    size_t vec_bytes        = n * sizeof(float);

    float *h_a = nullptr;
    float *h_b = nullptr;
    h_a = (float *)malloc(vec_bytes);
    h_b = (float *)malloc(vec_bytes);


    set_vec(h_a,n, Y);
    set_vec(h_b,n, Y_hat);




    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_a, vec_bytes); 
    cudaMalloc(&d_b, vec_bytes); 
    cudaMalloc(&d_c, vec_bytes); 
    cudaMalloc(&d_out, vec_bytes);
    float * h_out =(float *)malloc(vec_bytes);
    cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice);




    sub_vectors<<< blk_in_grid, THR_PER_BLK >>>(d_a,d_b,d_c, n);
    reduce6<THR_PER_BLK><<< blk_in_grid, THR_PER_BLK >>>(d_c, d_out,n);
    cudaMemcpy(h_out, d_out, vec_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    float out = h_out[0];
    free(h_out);
    return out;
}
float r2_score(float * Y, float * Y_hat, int n,float alpha){
    size_t vec_bytes        = n * sizeof(float);
	int blk_in_grid = ceil( float(n) / THR_PER_BLK );
    //need to accutally free memory here im being a bad bad boy
    float top = dif_squared(Y,Y_hat,n,blk_in_grid);
    float mean = get_vec_mean(Y,n,blk_in_grid);
    float *h_mean_vec = nullptr;
    h_mean_vec = (float *)malloc(vec_bytes);
    set_vec_val(h_mean_vec,n,mean); // more parallelization here I can do these concurrently the top and the bottom
    float bot = dif_squared(Y,h_mean_vec,n,blk_in_grid);
    free(h_mean_vec);
    return 1-(top/bot);
}


int main(void)
{
    
    // Error code to check return values for CUDA calls
    const int n = 1 <<3;
    const int m = 1<< 3;
    constexpr int N =  n;
    constexpr int M =  m;

    float matrix_A [n][m] ={};
    float matrix_B [n]= {};
    rand_arr<N,M>(matrix_A,n,m);
    rand_vec<N>(matrix_B,n);
    //make seperate ones for train and test 
    //we are going to chop different tau amounts this is probably done on cpu in parallel but the ridge and r2 on the cpu
    // will need to be careful because we will stop working with square matrices



    float  *h_weights = nullptr;    
    auto start = std::chrono::steady_clock::now();
    h_weights = (float *)malloc(n*sizeof(float));
    h_weights = ridge_reg<N,M>(matrix_A,matrix_B,n,m,0.0001);
    


    float * matrix_C = get_vec(vec_cuda_to_dev(XY(matrix_dev_to_cuda<N,M>(matrix_A,n),h_weights,n,n,1),n),n); 
    // this is getting our y hat
    //this gives me different answers at times, idk why 
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