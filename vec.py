from torch import Tensor
import torch
import numpy as np
import time; 
def ridge_fit_predict(
    X: Tensor,
    Y: Tensor,
    DEVICE: torch.device,
    eps: float = 1e-6,
) -> Tensor:
    """
    Ridge regression:

        w = (X^T X + alpha I)^(-1) X^T y

    Supports:
        ytr: [T] or [T, K] (multiple targets).

    Uses ONE solve with multiple RHS instead of one per target.
    See: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py
    """
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    if Y.dim() == 1:
        Y = Y.unsqueeze(1)   # [T, 1]

    T_train, n_feat = X.shape
    Xt = X.transpose(0, 1)     # [N, T]

    G = Xt @ X                 # [N, N]
    G = G + (eps) * torch.eye(
        n_feat, device=DEVICE, dtype=X.dtype
    )
    B = Xt @ Y                 # [N, K]

    L = torch.linalg.cholesky(G)
    W = torch.cholesky_solve(B, L)
    Y_hat = X @ W
    return corr2_score(y_true=Y,y_pred=Y_hat,eps=1e-12)



def corr2_score(y_true: Tensor, y_pred: Tensor, eps: float = 1e-12) -> float:
    """
    Squared Pearson correlation used in Jaeger-style memory capacity.
    """
    y_true_c = y_true - y_true.mean() ## remove the means
    y_pred_c = y_pred - y_pred.mean() ## remove the means 
    num = torch.sum(y_true_c * y_pred_c) ## this is top on pearson correlation
    den = torch.sqrt(torch.sum(y_true_c**2) * torch.sum(y_pred_c**2)) + eps
    corr2 = (num / den) ** 2
    return float(torch.clamp(corr2, 0.0, 1.0))



def random_vec(n):
    vec = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vec[i] = np.random.randint(1,100)
    return vec

def random_arr(n,m):
    vec = np.zeros((n,m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            vec[i][j] = np.random.randint(1,100)
    return vec

def main():
    n:int = 64
    m:int = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range (0,100):
        t = time.time_ns()
        mat = torch.from_numpy((random_arr(n,m)))
        vec = torch.from_numpy((random_vec(n)))
        ridge_fit_predict(mat,vec,device,1e-12)
        ##print((time.time_ns() - t) / 1_000_000)

main()