import numpy as np

_rng = np.random.default_rng(seed=2025)

def _power_method(
    A: np.ndarray,
    previous_eigenvectors: list[np.ndarray] | None = None,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:

    n = A.shape[0]
    if previous_eigenvectors is None:
        previous_eigenvectors = []

    v = _rng.standard_normal(n)

    v = _orthogonalize(v, previous_eigenvectors)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:

        v = _rng.standard_normal(n)
        v = _orthogonalize(v, previous_eigenvectors)
        norm_v = np.linalg.norm(v)
    v = v / norm_v

    eigenvalue = 0.0

    for iteration in range(max_iter):

        Av = A @ v

        if previous_eigenvectors and iteration % 5 == 0:
            Av = _orthogonalize(Av, previous_eigenvectors)

        norm_Av = np.linalg.norm(Av)
        if norm_Av < 1e-15:

            return 0.0, v

        v_new = Av / norm_Av

        eigenvalue_new = float(v_new @ (A @ v_new))

        if abs(eigenvalue_new - eigenvalue) < tol * max(abs(eigenvalue_new), 1.0):
            eigenvalue = eigenvalue_new
            v = v_new
            break
        eigenvalue = eigenvalue_new
        v = v_new

    if previous_eigenvectors:
        v = _orthogonalize(v, previous_eigenvectors)
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-15:
            v = v / norm_v

    return eigenvalue, v

def _orthogonalize(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:

    for _ in range(2):
        for u in basis:
            v = v - (u @ v) * u
    return v

def _eigen_decomposition(
    A: np.ndarray,
    num_eigenvalues: int,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:

    n = A.shape[0]
    eigenvalues = np.zeros(num_eigenvalues)
    eigenvectors = np.zeros((n, num_eigenvalues))

    A_deflated = A.copy().astype(np.float64)
    found_eigenvectors: list[np.ndarray] = []

    initial_norm = np.linalg.norm(A, 'fro')

    for i in range(num_eigenvalues):

        if i > 0 and initial_norm > 0:
            deflated_norm = np.linalg.norm(A_deflated, 'fro')
            if deflated_norm < tol * initial_norm:
                break

        eigenvalue, eigenvector = _power_method(
            A_deflated, found_eigenvectors, max_iter, tol
        )

        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
        found_eigenvectors.append(eigenvector.copy())

        A_deflated = A_deflated - eigenvalue * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors

def svd(
    X: np.ndarray, max_iter: int = 2000, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    X = X.astype(np.float64)
    m, n = X.shape
    k = min(m, n)

    C = X.T @ X
    eigenvalues_C, V = _eigen_decomposition(C, n, max_iter, tol)

    eigenvalues_C = np.maximum(eigenvalues_C, 0.0)
    all_sigma = np.sqrt(eigenvalues_C)
    sigma = all_sigma[:k]

    sv_max = sigma[0] if sigma[0] > 0 else 1.0
    sv_tol = max(m, n) * np.finfo(np.float64).eps * sv_max
    sv_tol = max(sv_tol, 1e-14)
    rank = int(np.sum(sigma > sv_tol))

    U_partial = np.zeros((m, rank))
    for i in range(rank):
        if sigma[i] > sv_tol:
            u_i = X @ V[:, i] / sigma[i]
        else:

            u_i = np.zeros(m)
        U_partial[:, i] = u_i

    for i in range(rank):
        for j in range(i):
            U_partial[:, i] -= (U_partial[:, j] @ U_partial[:, i]) * U_partial[:, j]
        norm_i = np.linalg.norm(U_partial[:, i])
        if norm_i > 1e-15:
            U_partial[:, i] /= norm_i

    U = np.zeros((m, m))
    U[:, :rank] = U_partial

    if m > rank:

        basis_count = rank
        attempts = 0
        while basis_count < m and attempts < m * 3:
            candidate = _rng.standard_normal(m)

            for b in range(basis_count):
                candidate -= (U[:, b] @ candidate) * U[:, b]

            for b in range(basis_count):
                candidate -= (U[:, b] @ candidate) * U[:, b]
            norm_c = np.linalg.norm(candidate)
            if norm_c > 1e-10:
                U[:, basis_count] = candidate / norm_c
                basis_count += 1
            attempts += 1

    Vt = V.T

    return U, sigma, Vt

def svd_compact(
    X: np.ndarray, max_iter: int = 2000, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    U_full, sigma_full, Vt_full = svd(X, max_iter, tol)

    sv_max = sigma_full[0] if len(sigma_full) > 0 and sigma_full[0] > 0 else 1.0
    sv_tol = max(X.shape) * np.finfo(np.float64).eps * sv_max
    sv_tol = max(sv_tol, 1e-14)
    rank = int(np.sum(sigma_full > sv_tol))
    if rank == 0:
        rank = 1

    return U_full[:, :rank], sigma_full[:rank], Vt_full[:rank, :]

def reconstruct(
    U: np.ndarray, sigma: np.ndarray, Vt: np.ndarray, k: int | None = None
) -> np.ndarray:

    if k is None:
        k = len(sigma)
    k = min(k, len(sigma))

    Sigma_k = np.diag(sigma[:k])
    return U[:, :k] @ Sigma_k @ Vt[:k, :]