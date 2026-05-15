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
    //printf("%f ",d_c[0]);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
           printf("%f ", d_c[j*n + i]);
            }
           printf("\n");
       }
        
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
    //print_vec(out,n*n,n);
    print_vec(out,n,std::sqrt(n));
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

// Converts row-major to column-major (for feeding INTO cublas)
float * set_arr_lin(int n,int m,float * arr){
    //https://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
    size_t bytes = (n*m)*sizeof(float);

    float * h_a = (float *)malloc(bytes);
    for (int i = 0; i < n*m; i++) {
            int row = i / m;
            int column = i %m; 
            h_a[column*n + row]= arr[i];
    }
    return h_a;
}

// Converts column-major to row-major (for reading FROM cublas)
float * set_arr_col_to_row(int n, int m, float * arr){ 
    size_t bytes = (n * m) * sizeof(float);
    float * h_a = (float *)malloc(bytes);
    
    for (int i = 0; i < n * m; i++) {
        int column = i / n; 
        int row = i % n;
        h_a[row * m + column] = arr[i];
    }
    return h_a;
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
    std::uniform_int_distribution<> dis(1, 5);
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
    std::uniform_int_distribution<> dis(1, 5);
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

// C = X - Y  (both in column-major, result in column-major)
float *XPY_s(cublasHandle_t handle, cudaStream_t stream,
             float *X, float *Y, int m, int n)
{
    const float alpha = 1.0f;
    const float beta  = -1.0f;
    size_t bytes = m * n * sizeof(float);
    float *d_c = nullptr;
    (cudaMalloc(&d_c, bytes));

    cublasSetStream(handle, stream);
    // CUBLAS_OP_N for both � no transpose needed if both are already column-major
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, &alpha, X, m, &beta, Y, m, d_c, m);
    return d_c;
}


// C = X * Y  (column-major)
// C is m x n, X is m x k, Y is k x n
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
        int start;
        View(float (&A_in)[n_s],int start, int bl, int r ,const int n_in){
            this->A = &A_in[start];
            this->start = start;
            this->block = bl;
            this->distance = n_in-bl;
            this->n=n_in;
            this->rows = r;
        }
        float get(int pos){
            int block_index = pos / this->block;
            int offset_in_block = pos % this->block;
            int final_index = (block_index * this->n) + offset_in_block;

            if (final_index >= 0 && final_index < ((this->n) * (this->n))-this->start) {
                return A[final_index];
            }
            else{
                printf("Error: Attempting to get value at out-of-bounds index %d\n", final_index);
                return 0;
            }
        }
        void print(){
            int num_vals= (this->rows) * (this->block);
            for(int i=0; i< num_vals;i++){
                if (i % this->block==0){
                    printf("\n");
                }
                printf(" %f ",this->get(i));
            }
        }
        // row-major flat array of the view's data
        float * get_for_cuda(){
            int num_vals= (this->rows) * (this->block);
            float * out = (float *)malloc(num_vals*sizeof(float));
            for(int i=0; i< num_vals;i++){
                out[i] = this->get(i);
            }
            return out;
        }
        // Sets row-major flat array
        void set(float * vals){
            int num_vals= (this->rows) * (this->block);
            for(int i=0; i< num_vals;i++){
                this->set_val(vals[i], i);
            } 
        }
        //  GPU memory row-major data
        void set_from_cuda(float * vals){
            int num_vals= (this->rows) * (this->block);
            size_t bytes = num_vals * sizeof(float);
            float * out =(float *)malloc(bytes);
            cudaMemcpy(out, vals, bytes, cudaMemcpyDeviceToHost);
            this->set(out);
            free(out);
        }
        // GREAT Graphics Proceessor Unit  COLUMN-MAJOR data (from cuBLAS)
        // from col-major [rows x block] to row-major before writing to View
        void set_from_cuda_col_major(float * vals){
            int num_vals= (this->rows) * (this->block);
            size_t bytes = num_vals * sizeof(float);
            float * col_data =(float *)malloc(bytes);
            cudaMemcpy(col_data, vals, bytes, cudaMemcpyDeviceToHost);
            // col-major: element (i,j) at index j*rows + i -> row-major: element (i,j) at index i*block + j
            float * row_data = (float *)malloc(bytes);
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->block; j++){
                    row_data[i * this->block + j] = col_data[j * this->rows + i];
                }
            }
            this->set(row_data);
            free(col_data);
            free(row_data);
        }

        void set_val(float val,int pos){
            int block_index = pos / this->block;
            int offset_in_block = pos % this->block;
            int final_index = (block_index * this->n) + offset_in_block;

            if (final_index >= 0 && final_index < ((this->n) * (this->n))-this->start) {
                this->A[final_index] = val;
            }
            else{
                printf("Error: Attempting to set value at out-of-bounds index %d\n", final_index);
            }
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
        else if (id ==6)starting with a clean vase, stripping lower leaves to prevent murky water, and cutting stems at a sharp angl
    {
        U[2] = 0;
    }
        else if (id ==7)
    {
        U[3] = a[3]-((a[2]/a[0])*a[1]);
    }
}

// solve_l: Solves X * U_11 = A_21 for L_21
__global__ void solve_l(float * U_11,float * a_21,float * L,int* bl,int* rows){
    int size_l = * rows;
    int block = *bl;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size_l*block){
        if (id % 2 == 0) { 
            L[id] = a_21[id]/U_11[0]; 
        } else {
            L[id] = (a_21[id]-((a_21[id-1]/U_11[0])*U_11[1]))/U_11[3]; 
        }
    }
}


// solve_u: Solves L_11 * X = A_12 for U_12
__global__ void solve_u(float * U,float * a_12, float * L_11,int* bl,int* rows){
    int size_l = * rows;    // = 2 (rows of a_12)
    int block = * bl;       // = num_cols (columns of a_12)
    int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < block*size_l){
        if (id < block) { 
            // first row: just copy
            U[id] = a_12[id];
        } else {
            // second row: U[1][j] = A_12[1][j] - A_12[0][j] * L_11[1][0]
            U[id] = (a_12[id]-((a_12[id-block]*L_11[2])));
        }
    }
}


__global__ void construct_final_lu(float * A, float * L, float * U, int n){    
	int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n*n){
        int row = id / n;
        int column = id %n; 
        
	if (row == column){
        L[id] = 1;
        U[id] = A[id];
        } else if (row < column){
        U[id] = A[id];
        L[id] = 0;
    }
    else{
        L[id] = A[id];
        U[id] = 0;
    }
    }
    }


template <int n_s>
void solve_block(View<n_s> a_11,View<n_s> a_21,View<n_s> a_12, View<n_s> a_22, int n,cublasHandle_t handle, cudaStream_t stream)
{
    // a_21 has shape [(n-bs-p) rows, 2 cols]  (block=2, rows=n-bs-p)
    // a_12 has shape [2 rows, (n-bs-p) cols]  (block=n-bs-p, rows=2)
    
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
    
    //print_cuda_vec(d_l,4);
        //print_cuda_vec(U,6);
        //print_cuda_vec(d_a_12,6);
    

    a_11.set_from_cuda(d_u);
    a_11.set_val(vec_cuda_to_dev(d_l,4)[2],2);
    
    if (a_21.rows*a_21.block > 0){

        // STEP 1: Compute L_21 = solve_l(U_11, A_21)
        // a_21 data is row-major [(n-bs-p) x 2], same as solve_l expects
        d_a = nullptr;
        size_t a_21_bytes = (a_21.rows * a_21.block) * sizeof(float);
        (cudaMalloc(&d_a, a_21_bytes));
        cudaMemcpy(d_a, a_21.get_for_cuda(), a_21_bytes, cudaMemcpyHostToDevice);
        
        int * d_block = nullptr;
        int * d_row = nullptr;
        float * L = nullptr;  // will hold L_21 result (row-major)
        float * U = nullptr;  // will hold U_12 result (row-major)
        
        cudaMalloc(&L, a_21_bytes);
        cudaMalloc(&d_block, sizeof(int));
        cudaMalloc(&d_row, sizeof(int));
        
        // For solve_l: block=2 (cols of a_21), rows=n-bs-p (rows of a_21)
        cudaMemcpy(d_block, &a_21.block, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row,   &a_21.rows, sizeof(int), cudaMemcpyHostToDevice);
        
        solve_l<<<BLK_IN_GRID, THR_PER_BLK>>>(d_u, d_a, L, d_block, d_row); 
        //printf("\n space \n");
	    //print_cuda_vec(L,4);
	    //printf("\n  \n");
	    //print_cuda_vec(d_u,4);
	    //printf("\n  \n");
	    //print_cuda_vec(d_a,4);
	    //printf("2\n");
	    
	
	    //printf("2\n");
	    //print_vec(a_21.get_for_cuda(),4);

      
        // Compute U_12 = solve_u(L_11, A_12)
        
        float * d_a_12 = nullptr;
        size_t a_12_bytes = (a_12.rows * a_12.block) * sizeof(float);
        (cudaMalloc(&d_a_12, a_12_bytes));
        cudaMalloc(&U, a_12_bytes);
        
        cudaMemcpy(d_a_12, a_12.get_for_cuda(), a_12_bytes, cudaMemcpyHostToDevice);
        
        // Update d_block and d_row for solve_u's needs
        cudaMemcpy(d_block, &a_12.block, sizeof(int), cudaMemcpyHostToDevice);  // = n-bs-p (num cols)
        cudaMemcpy(d_row,   &a_12.rows, sizeof(int), cudaMemcpyHostToDevice);   // = 2 (num rows)
        
        solve_u<<<BLK_IN_GRID, THR_PER_BLK>>>(U, d_a_12, d_l, d_block, d_row); 

        //  Write L_21 and U_12 back to matrix
        
        a_21.set_from_cuda(L);   
        a_12.set_from_cuda(U);   

        
        
        
        int remaining = a_21.rows;  // n - bs - pivot
        int bs = a_21.block;        // blocksize = 2
        
        // change  L_21 (row-major) to column-major for cuBLAS
        float *L_21_row = a_21.get_for_cuda();
        float *L_21_col = set_arr_lin(remaining, bs, L_21_row);
        float *L_21_cuda = vec_dev_to_cuda(L_21_col, remaining * bs);
        
        // Change U_12 (row-major) to column-major for cuBLAS
        float *U_12_row = a_12.get_for_cuda();
        float *U_12_col = set_arr_lin(bs, remaining, U_12_row);
        float *U_12_cuda = vec_dev_to_cuda(U_12_col, bs * remaining);
        
        
        float * xy = XY_s(handle, stream, L_21_cuda, U_12_cuda, remaining, remaining, bs);
        // xy is column-major [remaining x remaining]

        
        // STEP 5: A_22 = A_22 - L_21*U_12
        //Move paired commander  a_22 from row-major (View) to column-major (cuBLAS)
        float * d_a_22 = nullptr;
        size_t a_22_bytes = (a_22.rows * a_22.block) * sizeof(float);
        (cudaMalloc(&d_a_22, a_22_bytes));
        
        float *a_22_row = a_22.get_for_cuda();
        float *a_22_col = set_arr_lin(a_22.rows, a_22.block, a_22_row);
        cudaMemcpy(d_a_22, a_22_col, a_22_bytes, cudaMemcpyHostToDevice);
        free(a_22_row);
        free(a_22_col);
        
        float * xpy = XPY_s(handle, stream, d_a_22, xy, a_22.rows, a_22.block);
        // column-major [remaining x remaining]
        
        // view in row-major 
        a_22.set_from_cuda_col_major(xpy);
        
        // Cleanup
        free(L_21_row);
        free(L_21_col);
        free(U_12_row);
        free(U_12_col);
        cudaFree(d_a);
        cudaFree(L);
        cudaFree(U);
        cudaFree(d_block);
        cudaFree(d_row);
        cudaFree(d_a_12);
        cudaFree(L_21_cuda);
        cudaFree(U_12_cuda);
        cudaFree(xy);
        cudaFree(d_a_22);
        cudaFree(xpy);
    }
    
    cudaFree(d_l);
    cudaFree(d_u);
}


int main(void)
{

    //https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/main/linear-equations-3-lu-factorization-python.html

    const int n = 1 << 10;  // 8 (change to test different sizes)

    constexpr int N = n*n;

    float A[N] = {};

    auto start = std::chrono::steady_clock::now();
    rand_vec<N>(A, n*n);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);

    int blocksize = 2;
    for (int pivot = 0; pivot < n; pivot += 2){
            
        //print_vec(A, n*n, n);
        //printf("%f ",a.get(0));
	//printf("%f ",a.get(1));
	//printf("%f ",a.get(2));
        //printf("%f ",a.get(3));
        //printf("pivot=%d blocksize=%d n=%d\n", pivot, blocksize, n);
        
        View<N> a_11(A, (pivot*n)+pivot, blocksize, blocksize, n);
        View<N> a_22(A, ((pivot + blocksize) * n) + (pivot + blocksize), n-(blocksize+pivot), n-(blocksize+pivot), n);
        View<N> a_12(A, (pivot * n) + (pivot + blocksize), n-(blocksize+pivot), blocksize, n);
        View<N> a_21(A, ((pivot + blocksize) * n) + pivot, blocksize, n-(blocksize+pivot), n);
        
        solve_block<N>(a_11, a_21, a_12, a_22, n, handle, stream);
    }
    

    size_t bytes = n * n * sizeof(float);
    float  *d_L = nullptr;
    float  *d_U = nullptr;
    float *d_A = nullptr;
    cudaMalloc(&d_L, bytes);
    cudaMalloc(&d_U, bytes);
    cudaMalloc(&d_A, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

    construct_final_lu<<<BLK_IN_GRID, THR_PER_BLK>>>(d_A, d_L, d_U, n);
    //print out amazing results, take a victory lap, do a backflip. 
    printf("\nL:\n");
    //print_cuda_vec(d_L, n*n);
    printf("\nU:\n");
    //print_cuda_vec(d_U, n*n);
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("\n%lf ms\n", elapsed.count());
    
//clean up the mess and free shit up. Kids don't clean up the memory
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_A);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    printf("Done\n");
    return 0;
}