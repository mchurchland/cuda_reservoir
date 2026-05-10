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
nvcc -std=c++17 -dlto  -arch=sm_89 -I "$MATHDX/include"   -I "$MATHDX/external/cutlass/include"   shear.cu -o hi   "$MATHDX/lib/libcusolverdx.a"   -lcublas -lcusolver -lcurand

 */



void print_arr(float * d_c,int n,int m){
    //https://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
    for (int i = 0; i < n*m; i++) {
                    if (i%32==0){
                printf("\n");
            }
            printf("%d ", static_cast<int>(d_c[i]));

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
float * matrix_host_to_cuda(float h_[size_n][size_m],int n,int m){
    size_t bytes = n*m * sizeof(float);
    float  *d_ = nullptr;
        
    cudaMalloc(&d_, bytes);
    cudaMemcpy(d_, h_, bytes, cudaMemcpyHostToDevice);
    return d_;
}
template < int size_n, int size_m>
void rand_arr(float arr[size_n][size_m],int n,int m){
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Generator
    std::uniform_int_distribution<> dis(532,8211); // Range [1, 100]
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
    std::uniform_int_distribution<> dis(532,8211); // Range [1, 100]
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

__global__ void oet_step(float * c, bool * is_alt) // maybe I could als0o ahve something here for odd or even abool
{    /// AAAAAAA for even
    /// 5555555 for odd
	int id = blockDim.x * blockIdx.x + threadIdx.x;

    //could also consider doing it in a 3d array or like no
    // first I want to do an oet step
    // that means every thing odd idx should get smth from even index

    if (id < 1024){
    for (int i=0;i<16;i++){
    float x = c[id];
    if ((threadIdx.x / 32)%2 == 0 || * is_alt){
        //odds
        float y = __shfl_up_sync(0xAAAAAAAA, x, 1); // for odds im looking down
        if ((id-32)%32==0 && id !=0){
            y = c[id-32];
            if (x < y){ // then I swap these? // need to be careful abt the first idx
            //printf("%f %f\n",x,y);
            c[id] = y;
            c[id-32] = x;
        }
        }else{
        // I need to sort in snake like so like ya thats a pain in the ass
        if (x < y){ // then I swap these? // need to be careful abt the first idx
            //printf("%f %f\n",x,y);
            c[id] = y;
            c[id-1] = x;
        }   }
        //evens
        __syncthreads();
        x = c[id];
        y = __shfl_up_sync(0x55555555, x, 1); // for even im looking down
        
        // I need to sort in snake like so like ya thats a pain in the ass
        if (x < y){ // then I swap these? // need to be careful abt the first idx
            c[id] = y;
            c[id-1] = x;
        }
    } else {
        //odds i need to do the column but only for the odds
        float y = __shfl_up_sync(0xAAAAAAAA, x, 1); // for even im looking down
        if ((id-31)%32==0 && id !=0 && id-31 < 1024){
            
            
            y = c[id-32]; 

            if (x < y){ // then I swap these? // need to be careful abt the first idx
            //printf("%f %f\n",x,y);
            c[id] = y;
            c[id-32] = x;
        } } else {
        // I need to sort in snake like so like ya thats a pain in the ass
        

        if (x > y){ // then I swap these? // need to be careful abt the first idx
            //printf("%f %f\n",x,y);
            c[id] = y;
            c[id-1] = x;
        }   }
        //evens
        x = c[id];
        y = __shfl_up_sync(0x55555555, x, 1); // for even im looking down
        
        // I need to sort in snake like so like ya thats a pain in the ass
        if (x > y){ // then I swap these? // need to be careful abt the first idx
            c[id] = y;
            c[id-1] = x;
        }
    }}
}}




__global__ void transpose(float * c){
int id = blockDim.x * blockIdx.x + threadIdx.x;
int warp = id/32;
if (id > (32*warp)+(warp)){ // only work on the top side of the diag
    int i = id % 32;
    int j = id / 32;
    //printf("%d %d \n",i,j);
    int idx_to_swap = id+(31*(i-j));
    float x = c[id];
    c[id] = c[idx_to_swap]; 
    c[idx_to_swap] = x;
}
}




int main(void)
{
    //occasionally r2 score is low, check for racey stuff we can debug
    
    //test_xy();
    // Error code to check return values for CUDA calls
    const int n = 1 <<5; // dosent work for not nice cases // n cannot be greater than m
    const int m = 1 <<5;

    constexpr int N =  n;
    constexpr int M =  m;

    float matrix_A [n][m] ={};
    auto start = std::chrono::steady_clock::now();
    rand_arr<N,M>(matrix_A,n,m);
    bool  *d_bool = nullptr;
    bool a = false;
    cudaMalloc(&d_bool, sizeof(bool));
    
    float * cuda_mat = matrix_host_to_cuda<n,m>(matrix_A,n,m);
    print_cuda(cuda_mat,m,n);
    for(int i=0; i< 6; i++){
    a = false;
    cudaMemcpy(d_bool, &a, sizeof(bool), cudaMemcpyHostToDevice);
    oet_step<<< 1, 1024 >>>(cuda_mat, d_bool);
    //print_cuda(cuda_mat,m,3);
    //printf("\n trnaspose");
    transpose<<<16,1024>>>(cuda_mat);
    //print_cuda(cuda_mat,m,n);
    //printf("\n");
    a = true;
    cudaMemcpy(d_bool, &a, sizeof(bool), cudaMemcpyHostToDevice);
    oet_step<<< 1, 1024 >>>(cuda_mat, d_bool);
    transpose<<<16,1024>>>(cuda_mat);
    cudaPeekAtLastError();
}
     a = false;
    cudaMemcpy(d_bool, &a, sizeof(bool), cudaMemcpyHostToDevice);
    oet_step<<< 1, 1024 >>>(cuda_mat, d_bool);
    print_cuda(cuda_mat,m,n);
    auto end = std::chrono::steady_clock::now();
    
    // 3. Calculate the difference (duration)
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf( "Time elapsed: %lf\n", elapsed.count() );
    

    cudaDeviceReset();
    printf("Done\n");
    return 0;
}


