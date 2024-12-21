import numpy as np
from scipy.linalg import svd, inv

def singular_value_thresholding(matrix, tau):
    U, sigma, VT = svd(matrix, full_matrices=False)
    sigma_thresh = np.maximum(sigma - tau, 0)
    rank = np.sum(sigma_thresh > 0)
    return U[:, :rank] @ np.diag(sigma_thresh[:rank]) @ VT[:rank, :]

def latent_lrr(X, lambda_val):
    """
    Latent Low-Rank Representation for Subspace Segmentation and Feature Extraction
    Guangcan Liu, Shuicheng Yan. ICCV 2011.

    Solves:
        min_Z,L,E ||Z||_* + ||L||_* + lambda||E||_1,
        s.t. X = XZ + LX + E.

    Args:
        X: Input data matrix.
        lambda_val: Regularization parameter.

    Returns:
        Z: Low-rank representation.
        L: Latent low-rank component.
        E: Sparse error matrix.
    """
    A = X.copy()
    tol = 1e-6
    rho = 1.1
    max_mu = 1e6
    mu = 1e-6
    max_iter = int(1e6)

    d, n = X.shape
    m = A.shape[1]
    atx = X.T @ X
    inv_a = inv(A.T @ A + np.eye(m))
    inv_b = inv(A @ A.T + np.eye(d))

    # Initialize optimization variables
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    L = np.zeros((d, d))
    S = np.zeros((d, d))
    E = np.zeros((d, n))

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    Y3 = np.zeros((d, d))

    # Main optimization loop
    for iter in range(max_iter):
        # Update J using Singular Value Thresholding (SVT)
        temp_J = Z + Y2 / mu
        J = singular_value_thresholding(temp_J, 1 / mu)

        # Update S using Singular Value Thresholding (SVT)
        temp_S = L + Y3 / mu
        S = singular_value_thresholding(temp_S, 1 / mu)

        # Update Z
        Z = inv_a @ (atx - X.T @ L @ X - X.T @ E + J + (X.T @ Y1 - Y2) / mu)

        # Update L
        L = ((X - X @ Z - E) @ X.T + S + (Y1 @ X.T - Y3) / mu) @ inv_b

        # Update E
        xmaz = X - X @ Z - L @ X
        temp = xmaz + Y1 / mu
        E = np.maximum(0, temp - lambda_val / mu) + np.minimum(0, temp + lambda_val / mu)

        # Compute residuals
        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S

        max_l1 = np.max(np.abs(leq1))
        max_l2 = np.max(np.abs(leq2))
        max_l3 = np.max(np.abs(leq3))

        stop_criteria = max(max_l1, max(max_l2, max_l3))

        if stop_criteria < tol:
            print("LRR done.")
            break

        # Update Lagrange multipliers and penalty parameter
        Y1 += mu * leq1
        Y2 += mu * leq2
        Y3 += mu * leq3
        mu = min(max_mu, mu * rho)

    return Z, L, E
