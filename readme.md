I implemented various parts of ridge regression in varying levels of fidelity.
- Ridge Regression in CUDA
	- both using CUDA solverDX and the normal CUDA solver package
	- parallelized Pearson correlation coefficient loss
- Ridge Regression in python using tensors
- Blocked Lu decomposition in python
	- (only for block size 2)
	- this is close to what the cuda algorithms use for processing large non symmetrical positive definite arrays
- data_anal.py for analyzing the cuda data
- data.txt - data from cuda


#### Ridge Regression in CUDA
___
##### Helpers
- print_arr
	- prints an array on host thats in column major order
- print_vec
	- prints a vector on host
- print_cuda
	- prints a matrix on device
- print_cuda_vec
	- prints a vector on device
- vec_cuda_to_dev
	- moves a vector from device to host
- vec_dev_to_cuda
	- moves a vector from host to device
- set_arr
	- takes an array in row major and converts it to column
- set_vector
	- sets a vector
- set_vector_val
	- sets all values of a vector to a single value
- get_vector
	- makes an array and sets the values in it to that of a vector
- matrix_host_to_cuda
	- moves a matrix from the host to the device
- rand_arr
	- generates a float array with randomly set ints between 1 and 100 of size m x n
- rand vector
	- generates a float vector with randomly set ints between 1 and 100 of size n
##### CUBLAS/CUSOLVER code
- XTY
	- computes $X^T Y$
		- works for both square and rectangle matrices
- XY
	- computes $X Y$
		- works for both square and rectangle matrices
- XPY
	- computes $X+Y$
- XINV
	- Computes($X^{-1}$)
	- Solves using PIVOT
		- this helps when theres 0s on the diagonal,
	- Its not needed but was included for numerical stability
- Ridge Reg
	- Computes
		- $X^{T}X$
		- $X^{T}X + \alpha ID$
		- $(X^{T}X + \alpha ID)^{-1}$
		- $X^T Y$
		- $(X^{T}X + \alpha ID)^{-1}X^T Y$
	- You can solve either using CUDAsolver or CUDASOLVERDX

##### CUDASOLVERDX
- INV_DX
	- Computes($X^{-1}$)
	- Uses Cholsky solver (POSV)
	- Matrix must be hermitian positive definite for cholsky solver
	- It can solve matrices extreme quickly but can only do matrices up to 6x6 due to shared memory constraints
	- It uses the BLK_SIZE, and THR_PER_BLK as defined at the top of the file
	- Depending on the BLK_SIZE and THR_PER_BLK, it can fail leading to a matrix of Nans
		- I dont understand why this happens, it only happens rarely
		- I wondered if it was imprecision in floating point addition leading to non symmetrical matrices or problems with shared memory but it was not
		- I spent many hours debugging this and could not find any help online or in my testing, if I had unlimited time I would work on this

##### Self-implemented Kernels
- Reduce6
	- adjusted from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	- this does a very fast reduce of an input vector
- sub_val
	- subtracts a constant from all values in a matrix
- sub_val_sq
	- subtracts a constant from all values in a matrix
	- then element wise squares the matrix
- elementwise
	- element wise multiplication between two vectors
- mult
	- helper for element wise
- sum
	- helper for reduce6
- dif
	- helper for sub_val
	- helper for sub_val_sq

r2_score (this is pearson correlation squared)
- before calculating we subtract the mean of y from y and the mean of y_hat from y_hat
- Get_Top
	- helper for the numerator of pearson correlation squared
	- computes sum(mult(y,$\hat{y}$))
- Get_bot
	- helper for the denominator of pearson correlation squared
	- computes sqrt(sum($y^2$) $*$ sum$(\hat{y}^2$))
- Then it divides top by bottom and squares that value to get the pearson correlation coefficient squared.

This file computes
$W = (X^{T}X + \alpha ID)^{-1}X^T y$
$\hat{y} = X W$
Pearson$^2$ = ($\hat{y},y$) 
___
#### Ridge Regression in python using Numpy arrays for comparison
- ridge_fit_predict
	- does $(X^{T}X + \alpha ID)^{-1}X^T y$ on tensors
- corr2_score
	- Pearson$^2$ = ($\hat{y},y$) 
- random_vec
	- random vector of size n numbers between 1,100
- random_arr
	- random array of size n x m numbers between 1,100

___
#### Comparison between CUDA and PYTHON tensor code
(using data_anal.py)


| THR | BLK | mean time          | std                |
| --- | --- | ------------------ | ------------------ |
| 32  | 256 | 258.5719042        | 26.340566639320528 |
| 32  | 128 | 257.54859379999994 | 28.499192798172817 |
| 32  | 64  | 257.08336          | 26.478436393968554 |
| 64  | 256 | 256.02963845000005 | 25.9184727183783   |
| 64  | 128 | 257.7797734        | 27.582253538102112 |
| 64  | 64  | 257.1669129        | 28.712430752237122 |
| 128 | 256 | 257.3079565        | 25.643932827247237 |
| 128 | 128 | 257.05901439999997 | 26.544046968394277 |
| 128 | 64  | 257.64196000000004 | 27.381363089126335 |
| PY  | PY  | 804.83505475       | 41.90017200784613  |
The benchmark I ran consists of 100 random 6x6 float solves with time averaged over 20 trials.


There are two interesting results to observe from the benchmark run.
1. regardless of the THR_PER_BLK and BLK_SIZ, mean time stays remarkably consistent
	1. This suggests that the majority of run time is dominated by kernel launch / memory copying overhead, as its clear that allocating more computational resources does not impact the average speed of 100 solves
2. The Python script runs ~$3x$ slower than the CUDA script
	1. This is due to overhead from calling kernels in python
	2. Overhead from autograd, and dispatching
		1. source: https://discuss.pytorch.org/t/example-where-pytorch-substantially-slower-than-c-cuda/179076/5

My final project's main theme is overhead. It is clear that in working with solves on small matrices overhead is often times the largest cost in the computation. For CUDA, the overhead comes in the form of memory copies and kernel launches. For python it inherits these overheads and adds its own in the form of generalization in torch. My CUDA code assumes float values of a particular size. The torch code must handle a variety of datatypes. Its code may also involve more kernel launches or be optimized for larger matrix solves.

Speaking of larger matrices I wanted to investigate how GPU's can be used to solver larger matrices. These experiments elucidated the challenge of using shared memory for large matrix solves so I wanted to further investigate how that was done.


___
#### Deep Dive, stretch goal
Lu_decomp.py

I found that my CUDA methods only worked on up to 6x6 matrices

I wanted to understand how I could work on larger matrices, and how larger matrices are able to be done using global memory.

For this I investigated blocked LU decomposition

LU decomposition is the process to decomposing a matrix A into two parts: a lower triangular matrix L and an upper triangular matrix U.

Once a matrix is in LU form then solving is easy as one only would need to do back substitution.
source: https://www.cs.utexas.edu/~flame/laff/alaff/chapter05-launch.html
To do unblocked Lu decomposition you do this
![[Pasted image 20260510140110.png]]
To do the blocked update our $a_{11}$ is no longer a scalar but rather a blocksize X blocksize matrix

Thus: dimensions of each of our Matrix parts are as follows

- $A_{00}$ = pivot X pivot
- $a_{01}$ = pivot x blocksize
- $A_{02}$ = n-(pivot+blocksize) x pivot
- $a_{10}^T$ = blocksize x pivot 
- $\alpha_{11}$ = blocksize x blocksize
- $a_{12}^T$ = blocksize x n-(pivot+blocksize)
- $A_{20}$ = pivot x n-(pivot+blocksize)
- $a_{21}$ = n-(pivot+blocksize) X blocksize
- $A_{22}$ =  n-(pivot+blocksize) x  n-(pivot+blocksize)

Thus we have a function that breaks up the matrix A into its parts 
${A_{00}}{a_{01}}{A_{02}}{a_{10}^T}{\alpha_{11}}{a_{12}^T}{A_{20}}{a_{21}}{A_{22}}$


Furthermore we must do a more complicated update because $a_{11}$ is no longer a scalar in the blocked version.
![[Pasted image 20260510141125.png]]

Here is an example for how the matrix is broken up for 
n=6
blocksize =2
pivot = 2

| $A_{00}$   | $A_{00}$   | $A_{01}$ | $A_{01}$ | $A_{02}$   | $A_{02}$   |
| ---------- | ---------- | -------- | -------- | ---------- | ---------- |
| $A_{00}$   | $A_{00}$   | $A_{01}$ | $A_{01}$ | $A_{02}$   | $A_{02}$   |
| $A_{10}^T$ | $A_{10}^T$ | $A_{11}$ | $A_{11}$ | $A_{12}^T$ | $A_{12}^T$ |
| $A_{10}^T$ | $A_{10}^T$ | $A_{11}$ | $A_{11}$ | $A_{12}^T$ | $A_{12}^T$ |
| $A_{20}$   | $A_{20}$   | $A_{21}$ | $A_{21}$ | $A_{22}$   | $A_{22}$   |
| $A_{20}$   | $A_{20}$   | $A_{21}$ | $A_{21}$ | $A_{22}$   | $A_{22}$   |


Thus we use the schur complement to update $A_{22}$

We follow these steps seen [here](https://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec09.pdf)
![[Pasted image 20260510141403.png]]

First we solve and update $A_{11}$ by doing a simple 2x2 solve (seen in solve_block_2x2)
This gives us both $L_{11},U_{11}$
Then we compute $L_{21},U_{12}$ using the functions solve_l and solve_u, these are simple back substitution solves. In the code itself I denoted what could be parallelized and what must be done sequentially.

Once these are computed we do a matrix multiplication and subtract from $A_{22}$ 

We update the matrix with
$A_{11} = solved(A_{11})$
$A_{22} = L_{21}U_{12}$
$A_{21}$ = $L_{21}$
$A_{12} = U_{12}$
We repeat until the blocks reach the end at which point we have a factored matrix.
Here are the steps that are repeated
- breakup the matrix
- get updated $A_{11}$ and $L_{21}U_{12}$
- update $A_{22}$
- update matrix
Once this has finished we break the matrix apart into the L and U parts, observe that $LU$ is equal to the original matrix $A$

#### Commentary on LU decomp
___
This code has lots of opportunities for parallelism and I have commented in the code all of the ways I could think of to parallelize the code. I have commented where operations should be done in parallel in order to improve wall clock time performance.

I should note that this code allows one to solve matrices while not loading the whole matrix into shared memory. The amount of shared memory used by this program scales super linearly with blocksize. And (except for $A_{22}$, also depends on how matrix multiplication is implemented) linearly with matrix size. The only parts of the matrix needed in shared memory are $A_{11},A_{22},A_{21},A_{12}$ and $L_{21},U_{12}$. The size of these matrices is mainly controlled by the block size. Furthermore, if the matrix is so big that one or all of these cannot fit into shared memory each of the steps can be batched so that only a smaller portion of the matrix is in shared memory. 
