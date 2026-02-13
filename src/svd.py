"""
SVD (Singular Value Decomposition) implementata da zero in Python.

Fattorizza una matrice X in X = U · Σ · Vᵀ dove:
- U: matrice con colonne ortonormali (vettori singolari sinistri)
- Σ: matrice diagonale con i valori singolari in ordine decrescente
- V: matrice con colonne ortonormali (vettori singolari destri)

L'algoritmo utilizza il metodo delle potenze con deflazione per calcolare
autovalori e autovettori delle matrici XᵀX (per V) e XXᵀ (per U).
"""

import numpy as np


# RNG globale per il modulo (non usa seed fisso, così ogni chiamata è indipendente)
_rng = np.random.default_rng(seed=2025)


def _power_method(
    A: np.ndarray,
    previous_eigenvectors: list[np.ndarray] | None = None,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """
    Metodo delle potenze per trovare l'autovalore dominante e il corrispondente
    autovettore di una matrice simmetrica A.

    Include ri-ortogonalizzazione rispetto agli autovettori già trovati
    per gestire correttamente autovalori ripetuti e ridurre l'errore numerico.

    Parametri
    ---------
    A : np.ndarray
        Matrice simmetrica (n×n).
    previous_eigenvectors : list[np.ndarray], opzionale
        Autovettori già calcolati, per l'ortogonalizzazione.
    max_iter : int
        Numero massimo di iterazioni.
    tol : float
        Soglia di convergenza sulla variazione relativa dell'autovalore.

    Ritorna
    -------
    eigenvalue : float
        L'autovalore dominante.
    eigenvector : np.ndarray
        L'autovettore unitario corrispondente.
    """
    n = A.shape[0]
    if previous_eigenvectors is None:
        previous_eigenvectors = []

    # Inizializzazione con un vettore casuale diverso ogni volta
    v = _rng.standard_normal(n)

    # Ortogonalizza rispetto agli autovettori già trovati
    v = _orthogonalize(v, previous_eigenvectors)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        # Il vettore random era nello span degli autovettori precedenti, riprova
        v = _rng.standard_normal(n)
        v = _orthogonalize(v, previous_eigenvectors)
        norm_v = np.linalg.norm(v)
    v = v / norm_v

    eigenvalue = 0.0

    for iteration in range(max_iter):
        # Moltiplicazione matrice-vettore
        Av = A @ v

        # Ri-ortogonalizza periodicamente per stabilità numerica
        if previous_eigenvectors and iteration % 5 == 0:
            Av = _orthogonalize(Av, previous_eigenvectors)

        # Normalizzazione
        norm_Av = np.linalg.norm(Av)
        if norm_Av < 1e-15:
            # La matrice ha mandato il vettore a zero → autovalore nullo
            return 0.0, v

        v_new = Av / norm_Av

        # Stima dell'autovalore (quoziente di Rayleigh)
        eigenvalue_new = float(v_new @ (A @ v_new))

        # Controllo di convergenza
        if abs(eigenvalue_new - eigenvalue) < tol * max(abs(eigenvalue_new), 1.0):
            eigenvalue = eigenvalue_new
            v = v_new
            break
        eigenvalue = eigenvalue_new
        v = v_new

    # Ortogonalizzazione finale
    if previous_eigenvectors:
        v = _orthogonalize(v, previous_eigenvectors)
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-15:
            v = v / norm_v

    return eigenvalue, v


def _orthogonalize(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """
    Ortogonalizza il vettore v rispetto a una lista di vettori ortonormali (Gram-Schmidt).
    Esegue due passi per migliorare la stabilità numerica (re-orthogonalization).
    """
    for _ in range(2):  # Due passi di Gram-Schmidt per stabilità
        for u in basis:
            v = v - (u @ v) * u
    return v


def _eigen_decomposition(
    A: np.ndarray,
    num_eigenvalues: int,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola i primi `num_eigenvalues` autovalori e autovettori di una matrice
    simmetrica A usando il metodo delle potenze con deflazione.

    Parametri
    ---------
    A : np.ndarray
        Matrice simmetrica (n×n).
    num_eigenvalues : int
        Numero di autovalori/autovettori da calcolare.
    max_iter : int
        Iterazioni massime per ogni chiamata al metodo delle potenze.
    tol : float
        Soglia di convergenza.

    Ritorna
    -------
    eigenvalues : np.ndarray
        Array di autovalori in ordine decrescente.
    eigenvectors : np.ndarray
        Matrice (n × num_eigenvalues) le cui colonne sono gli autovettori.
    """
    n = A.shape[0]
    eigenvalues = np.zeros(num_eigenvalues)
    eigenvectors = np.zeros((n, num_eigenvalues))

    A_deflated = A.copy().astype(np.float64)
    found_eigenvectors: list[np.ndarray] = []

    for i in range(num_eigenvalues):
        eigenvalue, eigenvector = _power_method(
            A_deflated, found_eigenvectors, max_iter, tol
        )

        # Salva autovalore e autovettore
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
        found_eigenvectors.append(eigenvector.copy())

        # Deflazione: rimuove la componente dell'autovalore trovato
        # A_new = A_old - λ * v * vᵀ
        A_deflated = A_deflated - eigenvalue * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors


def svd(
    X: np.ndarray, max_iter: int = 2000, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular Value Decomposition (SVD) implementata da zero.
    
    Fattorizza la matrice X (m×n) in X = U · Σ · Vᵀ.

    Algoritmo:
    1. Calcola la matrice di covarianza C = XᵀX
    2. Trova gli autovettori di C → colonne di V (direzioni principali)
    3. Calcola i valori singolari σᵢ = √(λᵢ)
    4. Calcola U = X · V · Σ⁻¹
    5. Gestisce eventuali componenti nulle del null-space

    Parametri
    ---------
    X : np.ndarray
        Matrice di input (m×n).
    max_iter : int
        Iterazioni massime per il metodo delle potenze.
    tol : float
        Soglia di convergenza.

    Ritorna
    -------
    U : np.ndarray
        Matrice (m×m) con colonne ortonormali (vettori singolari sinistri).
    sigma : np.ndarray
        Vettore dei valori singolari in ordine decrescente (lunghezza min(m,n)).
    Vt : np.ndarray
        Matrice (n×n) → Vᵀ (vettori singolari destri, trasposta).
    """
    X = X.astype(np.float64)
    m, n = X.shape
    k = min(m, n)  # numero massimo di valori singolari non nulli

    # ──────────────────────────────────────────────
    # Passo 1-2: Calcolo di V tramite la matrice XᵀX
    # ──────────────────────────────────────────────
    # La matrice di covarianza (proporzionale) C = XᵀX è (n×n).
    # I suoi autovettori sono le colonne di V.
    C = X.T @ X  # (n×n)
    eigenvalues_C, V = _eigen_decomposition(C, n, max_iter, tol)

    # ──────────────────────────────────────────────
    # Passo 3: Calcolo dei valori singolari
    # ──────────────────────────────────────────────
    # σᵢ = √(λᵢ), con λᵢ autovalori di XᵀX.
    # Forziamo a zero eventuali valori negativi dovuti a errori numerici.
    eigenvalues_C = np.maximum(eigenvalues_C, 0.0)
    all_sigma = np.sqrt(eigenvalues_C)
    sigma = all_sigma[:k]  # solo i primi min(m,n)

    # ──────────────────────────────────────────────
    # Passo 4: Calcolo di U
    # ──────────────────────────────────────────────
    # Per i valori singolari non nulli: uᵢ = (1/σᵢ) · X · vᵢ
    
    # Determina il rango numerico (soglia più robusta)
    sv_tol = max(m, n) * np.finfo(np.float64).eps * (sigma[0] if sigma[0] > 0 else 1.0)
    rank = int(np.sum(sigma > sv_tol))

    # Calcola le prime `rank` colonne di U usando U = X · V · Σ⁻¹
    U_partial = np.zeros((m, rank))
    for i in range(rank):
        u_i = X @ V[:, i] / sigma[i]
        U_partial[:, i] = u_i

    # Ri-ortogonalizza U_partial con Gram-Schmidt per stabilità
    for i in range(rank):
        for j in range(i):
            U_partial[:, i] -= (U_partial[:, j] @ U_partial[:, i]) * U_partial[:, j]
        norm_i = np.linalg.norm(U_partial[:, i])
        if norm_i > 1e-15:
            U_partial[:, i] /= norm_i

    # ──────────────────────────────────────────────
    # Passo 4b: Completamento di U per la full SVD (m×m)
    # ──────────────────────────────────────────────
    U = np.zeros((m, m))
    U[:, :rank] = U_partial

    if m > rank:
        # Completa con vettori random ortogonalizzati (null-space di Xᵀ)
        basis_count = rank
        attempts = 0
        while basis_count < m and attempts < m * 3:
            candidate = _rng.standard_normal(m)
            # Ortogonalizza rispetto a tutte le colonne già presenti
            for b in range(basis_count):
                candidate -= (U[:, b] @ candidate) * U[:, b]
            # Secondo passo Gram-Schmidt
            for b in range(basis_count):
                candidate -= (U[:, b] @ candidate) * U[:, b]
            norm_c = np.linalg.norm(candidate)
            if norm_c > 1e-10:
                U[:, basis_count] = candidate / norm_c
                basis_count += 1
            attempts += 1

    # ──────────────────────────────────────────────
    # V è già (n×n) perché abbiamo calcolato n autovettori di C
    # ──────────────────────────────────────────────
    Vt = V.T

    return U, sigma, Vt


def svd_compact(
    X: np.ndarray, max_iter: int = 2000, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD in forma compatta (economy/thin SVD).
    
    Ritorna solo le componenti relative ai valori singolari non nulli,
    utile per la ricostruzione e per la steganografia.

    Parametri
    ---------
    X : np.ndarray
        Matrice di input (m×n).

    Ritorna
    -------
    U : np.ndarray
        Matrice (m×r) dove r = rank(X).
    sigma : np.ndarray
        Vettore dei valori singolari non nulli, in ordine decrescente.
    Vt : np.ndarray
        Matrice (r×n).
    """
    U_full, sigma_full, Vt_full = svd(X, max_iter, tol)

    # Determina il rango con soglia robusta
    sv_tol = max(X.shape) * np.finfo(np.float64).eps * (sigma_full[0] if len(sigma_full) > 0 and sigma_full[0] > 0 else 1.0)
    rank = int(np.sum(sigma_full > sv_tol))
    if rank == 0:
        rank = 1  # Almeno un valore singolare

    return U_full[:, :rank], sigma_full[:rank], Vt_full[:rank, :]


def reconstruct(
    U: np.ndarray, sigma: np.ndarray, Vt: np.ndarray, k: int | None = None
) -> np.ndarray:
    """
    Ricostruisce la matrice X (o una sua approssimazione di rango k)
    a partire dalla decomposizione SVD.
    
    X_approx = U[:, :k] · diag(σ[:k]) · Vᵀ[:k, :]

    Parametri
    ---------
    U : np.ndarray
        Matrice dei vettori singolari sinistri.
    sigma : np.ndarray
        Vettore dei valori singolari.
    Vt : np.ndarray
        Matrice dei vettori singolari destri (trasposta).
    k : int, opzionale
        Numero di valori singolari da utilizzare. Se None, usa tutti.

    Ritorna
    -------
    X_reconstructed : np.ndarray
        Matrice ricostruita.
    """
    if k is None:
        k = len(sigma)
    k = min(k, len(sigma))

    # X_approx = U[:, :k] · diag(σ[:k]) · Vᵀ[:k, :]
    Sigma_k = np.diag(sigma[:k])
    return U[:, :k] @ Sigma_k @ Vt[:k, :]
