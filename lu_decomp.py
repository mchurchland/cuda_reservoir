import numpy as np
def split_matrix(A,pivot_col,blocksize,n):
  assert pivot_col %blocksize ==0, "incorrect pivot col"

  A_00 = A[0:pivot_col,0:pivot_col] ## these can all be done in parallel so happy yay
  a_11 = A[pivot_col:pivot_col+blocksize,pivot_col:pivot_col+blocksize]
  a_01 = A[0:pivot_col,pivot_col:pivot_col+blocksize]
  a_10_t = A[pivot_col:pivot_col+blocksize,0:pivot_col]
  A_22 = A[blocksize+pivot_col:,blocksize+pivot_col:]
  A_02 = A[0:pivot_col,pivot_col+blocksize:]
  A_20 = A[pivot_col+blocksize:,0:pivot_col]
  a_12_t = A[pivot_col:pivot_col+blocksize,blocksize+pivot_col:]
  a_21 = A[blocksize+pivot_col:,pivot_col:pivot_col+blocksize]

  return A_00,a_11,a_01,a_10_t,A_22,A_02,A_20,a_12_t,a_21

def update_a22(A_22,a_21,a_12_t):
  A_22 -= a_21 @ a_12_t

def hard_update(A_11,A_21,A_12): ## right now this only works for 2x2
  L_11,U_11 = solve_block_2x2(A_11) ## easy solve
  ## now we need L_21 
  ##compute X U_11 =  A_21 to solve for L_21
  L_21 = solve_l(U_11,A_21) ## These can happen concurrently

  ## we need to find U_12
  ## compute L_11 X    = A_12 _t to solve for U_12
  U_12 = solve_u(L_11,A_12) ## These can happen concurrently
  
  out_A_11 = U_11 ## assuming block size 2
  out_A_11[1][0] = L_11[1,0]
  return L_21,U_12, out_A_11


def solve_l(U_11,A_21): ## derived
  size_of_L = len(A_21)
  L= np.zeros((size_of_L,2))
  for idx,row in enumerate(A_21): ## each is indep
    L[idx][0] = A_21[idx][0]/U_11[0][0] ## these must be done sequentially ## this oculd be done in parallel depending on if you system is processor or memory bottlenecked
    ## if it is memory bottle necked then do them in parallel because the cost to recompute A_21[idx][0]/U_11[0][0] is less than the cost to get it from mem
    L[idx][1] = (A_21[idx][1]-(L[idx][0]*U_11[0][1]))/U_11[1][1]
  return L

def solve_u(L_11,A_12): ## derived
  size_of_U = len(A_12[0])
  U= np.zeros((2,size_of_U))
  for idx,row in enumerate(A_12[0]): ## each is indep
    U[0][idx] = A_12[0][idx] ## can be done in parallel
    U[1][idx] = (A_12[1][idx]-(A_12[0][idx]*L_11[1][0])) ## A_12[idx][0] =  U[idx][0] so this can be done in parallel
  return U

"""
U_11 = np.array([[2,1],[0,3]])
target = np.array([[2,2],[1,1]])
solve_l(U_11,target)

U_11 = np.array([[1,0],[2,1]])
target = np.array([[2,2],[1,1]])
solve_u(U_11,target)"""

""
def solve_block_2x2(A): ## this is for 2x2 
  ## A = [a  b
  ##      c  d]
  ## L = [ 1 0
  ##     c/a 1]
  ## U = [a    b
  ##      0    d-(c/a)b]
  assert A.size ==4
  L = np.zeros((2,2)) ## allocations in parallel
  U = np.zeros((2,2))
  ##done in parallel
  L[0][0] = 1
  L[0][1] = 0
  L[1][0] = A[1][0] / A[0][0] ## these accesses can be done in parallel
  L[1][1] = 1

  U[0][0] = A[0][0] ## these can all be done in parallel
  U[0][1] = A[0][1]
  U[1][0] = 0
  U[1][1] = A[1][1] - ((A[1][0]/A[0][0]) * A[0][1])
  return L,U 

def update_mat(A,pivot_col,blocksize,n,a_12,a_21,a_11,a_22):
  ## we gotta update A_12 and A_21 with their updated values which and l and u
  ## gotta replace a_11
  ## gotta replace a_22
  ## these are all parallel
  A[pivot_col:pivot_col+blocksize,pivot_col:pivot_col+blocksize] = a_11
  A[blocksize+pivot_col:,blocksize+pivot_col:] = a_22
  A[pivot_col:pivot_col+blocksize,blocksize+pivot_col:] = a_21
  A[blocksize+pivot_col:,pivot_col:pivot_col+blocksize] = a_12

def construct_final_lu(A,n):
  L = np.zeros((n,n))
  U = np.zeros((n,n))
  for i in range(0,n): ## super easy to parallelize O(1) lol
    for j in range(0,n): ##cuda would love to do this 
      if i==j:
        L[i][j] =1
      if i<=j:
        U[i][j] = A[i][j]
      else:
        L[i][j] = A[i][j]
  return L,U

n=100
A = np.random.randint(1, 10, size=(n, n)).astype('float64')

print(A)
for p in range(0,n,2):
  ##print(A)
  A_00,a_11,a_01,a_10_t,A_22,A_02,A_20,a_12_t,a_21 = split_matrix(A,p,2,n) ## I think the u and t are 
  L_21,U_12,a_11  = hard_update(a_11,a_21,a_12_t) ## block update step https://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec09.pdf

  update_a22(A_22,L_21,U_12) ##sequential
  update_mat(A,p,2,n,L_21,U_12,a_11,A_22) ##sequential

L, U = construct_final_lu(A, n)
print(L)
print(U)
print(L@U)

## shared memory scales superlinearly with block size