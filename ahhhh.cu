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
#include <cassert>
#define THR_PER_BLK 64
#define BLK_IN_GRID 64
/**
cublas does column major order
conda activate cuda13
MATHDX=/home/milesc/para/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03  
nvcc -std=c++17 -dlto  -arch=sm_89 -I "$MATHDX/include"   -I "$MATHDX/external/cutlass/include"   ahhhh.cu -o hi   "$MATHDX/lib/libcusolverdx.a"   -lcublas -lcusolver -lcurand

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




void print_vec(float * d_c,int n,int m){
    for (int i = 0; i < n; i++) {
                    if (i%m==0 && i!=0){
                printf("\n");
            }
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
    print_vec(out,n*n,n);
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
    std::uniform_int_distribution<> dis(-2, 2); // Range [1, 100] doing 1,100 is problematic and leads to solves not working
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
    std::uniform_int_distribution<> dis(-2, 2); // Range [1, 100] doing 1,100 is problematic and leads to solves not working, a google search about
    // the probability of a 1,100 sampled matrix lead me to this, it helped me realize that the problem is not the distirubtion but the **amplification**
    // with 1,100 the floats get super large (about 2/100 times) in xtx and thus lead to problems
    // it turns out that Floats are generally only reliable to 6-7 significant decimal digits. and thus here occasionally break when the numbers in the matrix are large
    // having an even distribution changes this as the negative numbers "dampen" the amplification during xtx
    // google’s built-in AI summary helped me here
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


float *XY_s(cublasHandle_t handle, cudaStream_t stream,
            float *X, float *Y, int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    size_t bytes = m * n * sizeof(float);
    float *d_c = nullptr;
    (cudaMalloc(&d_c, bytes));

    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, &alpha, X, m, Y, k, &beta, d_c, m);
    return d_c;
}



/**
 * Host main routine
 */





template <int n_s>
class View{ //assume that bl is the blk size of the cols, ie we have this many rows, ill play losey gosey with the columns and (stupidly) trust myself
    public:
        float * A;
        int block;
        int distance;
        int n;
        int rows;
        View(float  A_in[n_s],int start, int bl, int r ,const int n_in){
            this->A = &A_in[start];
            this->block = bl;
            this->distance = n_in-bl;
            this->n=n_in;
            this->rows = r;
        }
        __host__ __device__ float get(int pos){
            if (pos==0){
                return A[0];
            }else{      
            assert ( static_cast<int>(std::floor(static_cast<float>(pos) /this->block)) < this->rows);
            //printf("\n%d \n",(static_cast<int>(std::floor(static_cast<float>(pos) /this->block)) * this->n+  pos%this->block));
            return this->A[(static_cast<int>(std::floor(static_cast<float>(pos) /this->block)) * this->n+  pos%this->block)];
        }}
        void print(){
            int num_vals= (this->rows) * (this->block);
            for(int i=0; i< num_vals;i++){
                if (i % this->block==0){
                    printf("\n");
                }
                printf(" %f ",this->get(i));
            }
        }
        float * get_for_cuda(){
            
            int num_vals= (this->rows) * (this->block);
            float * out = (float *)malloc(num_vals*sizeof(float));
            for(int i=0; i< num_vals;i++){
                out[i] = this->get(i);
            }
            return out;
    }
            
};


__global__ void solve_block_2x2(float * a, float * L,float* U)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id ==0){
        L[0]= 1;
    }
    else if (id ==1)
    {
        L[1] = 0;
    }
        else if (id ==2)
    {
        L[2] = a[2]/a[0];
    }
        else if (id ==3)
    {
        L[3] = 1;
    }
        else if (id ==4)
    {
        U[0] = a[0];
    }
        else if (id ==5)
    {
        U[1] = a[1];
    }
        else if (id ==6)
    {
        U[2] = 0;
    }
        else if (id ==7)
    {
        U[3] = a[3]-((a[2]/a[0]*a[2]));
    }
}

__global__ void solve_l(float * U_11,float * a_21,float * L,int* bl,int* rows){
    //ideally a_21 is in shared memory

    int size_l = * rows;
    int block = * bl;
    	int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size_l){
        
    if (id %2==0)                { 
        
        L[id] = a_21[id]/U_11[0]; //sexy ash
    } else{
        printf("\n%d %f\n",id,(a_21[id-1]/U_11[0]));
        L[id] = (a_21[id]-((a_21[id-1]/U_11[0])*U_11[1]))/U_11[3]; //sexy ash
    }
}
}


__global__ void solve_u(float * U_11,float * a_21){

}


__global__ void construct_final_lu(float * A, float * L, float * U, int n){    
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id <= n*n){
        int row = id / n;
        int column = id %n; 
	if (row == column){
        L[id] = 1;
        } else if (row < column){
        U[id] = A[id];
    }
    else{
        L[id] = A[id];
    }
    }
    }
typedef struct {
    float *d_l;
    float *d_u;
} PointerPair;

template <int n_s>
PointerPair solve_block(cudaStream_t stream, View<n_s> a_11,View<n_s> a_21, int n)
{
    size_t bytes = 2 * 2 * sizeof(float);

    float *d_a = nullptr;
    float *d_l = nullptr;
    float *d_u = nullptr;

    (cudaMalloc(&d_a, bytes));
    (cudaMalloc(&d_l, bytes));
    (cudaMalloc(&d_u, bytes));
    cudaMemcpy(d_a, a_11.get_for_cuda(), bytes, cudaMemcpyHostToDevice);
    solve_block_2x2<<<BLK_IN_GRID, THR_PER_BLK>>>(d_a,d_l,d_u);
    cudaFree(d_a);
    PointerPair pairs;
    pairs.d_l = d_l;
    pairs.d_u = d_u;
    d_a = nullptr;
    size_t a_21_bytes = (a_21.rows*a_21.block)*sizeof(float);
    (cudaMalloc(&d_a, a_21_bytes));

    cudaMemcpy(d_a, a_21.get_for_cuda(), a_21_bytes, cudaMemcpyHostToDevice);
    int * d_block = nullptr;
    int * d_row = nullptr;
    float * L = nullptr;
    cudaMalloc(&L, a_21_bytes);
    cudaMalloc(&d_block, sizeof(int));
    cudaMalloc(&d_row, sizeof(int));

   
    cudaMemcpy(d_block, &a_21.block, sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_row,   &a_21.rows, sizeof(int),cudaMemcpyHostToDevice);


    solve_l<<<BLK_IN_GRID, THR_PER_BLK>>>(d_u,d_a,L,d_block,d_row); 
    print_cuda_vec(d_u,4);
        print_cuda_vec(d_a,4);

    print_cuda_vec(L,4);
    return pairs;
}

template <int n_s>
void split_matrix(float  A[n_s],int pivot,int blocksize,int n){
      ///   m sta , cols, rows, 
    View<n_s> a_00(A,0,pivot,pivot,n);
    View<n_s> a_11(A,(pivot*n)+pivot,blocksize,blocksize,n);
    View<n_s> a_01(A,pivot,blocksize,pivot,n);
    View<n_s> a_10(A,(pivot*n),pivot,blocksize,n);
    View<n_s> a_22(A,((pivot*n)+(blocksize*n)+blocksize*2),n-(blocksize+pivot),n-(blocksize+pivot),n); // this might be blocksize squared for the general case
    //shouldnt this be two columns not 4
    View<n_s> a_02(A,pivot+blocksize,n-(blocksize+pivot),pivot,n); 
    View<n_s> a_20(A,(pivot*n)+(blocksize*n),pivot,n-(blocksize+pivot),n); // this might be blocksize squared for the general case
    View<n_s> a_12(A,(pivot*n)+(blocksize*2),n-(blocksize+pivot),blocksize,n); // this might be blocksize squared for the general case
    View<n_s> a_21(A,(pivot*n)+(blocksize*n)+blocksize,blocksize,n-(blocksize+pivot),n); // this might be blocksize squared for the general case    

} 


//update_22 matrix add and subtract

//solve L for 2x2
//solve u for 2x2 kenel

//solve 2x2 kernel


int main(void)
{

    //https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/main/linear-equations-3-lu-factorization-python.html

    //this is it for 4x4
    const int n = 1 << 3;  // 64

    constexpr int N = n;
    cudaStream_t streams;
    cudaStreamCreateWithFlags(&streams, cudaStreamNonBlocking);

    float A[n] = {};

    //for (int j = 0; j < 1; j++) {
        auto start = std::chrono::steady_clock::now();
        //for (int i = 0; i < 1; i++) {
            rand_vec<N>(A, n*n);
            int pivot = 2;
            int blocksize =2;
            //print_vec(A,n*n,n);
            //printf("%f ",a.get(0));
            //printf("%f ",a.get(1));
            //printf("%f ",a.get(2));
            //printf("%f ",a.get(3));
            //split_matrix<N>(matrix_B,2,2,n);
            View<N> a_00(A,0,pivot,pivot,n);
            View<N> a_11(A,(pivot*n)+pivot,blocksize,blocksize,n);
            View<N> a_01(A,pivot,blocksize,pivot,n);
            View<N> a_10(A,(pivot*n),pivot,blocksize,n);
            View<N> a_22(A,((pivot*n)+(blocksize*n)+blocksize*2),n-(blocksize+pivot),n-(blocksize+pivot),n); // this might be blocksize squared for the general case
            //shouldnt this be two columns not 4
            View<N> a_02(A,pivot+blocksize,n-(blocksize+pivot),pivot,n); 
            View<N> a_20(A,(pivot*n)+(blocksize*n),pivot,n-(blocksize+pivot),n); // this might be blocksize squared for the general case
            View<N> a_12(A,(pivot*n)+(blocksize*2),n-(blocksize+pivot),blocksize,n); // this might be blocksize squared for the general case
            View<N> a_21(A,(pivot*n)+(blocksize*n)+blocksize,blocksize,n-(blocksize+pivot),n); // this might be blocksize squared for the general case    
            

            solve_block<N>(streams,a_11,a_22,n);

            //printf("%f ",a.get(4));

            
        //}

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("\n%lf\n", elapsed.count());
  //  }

    printf("Done\n");
    return 0;
}