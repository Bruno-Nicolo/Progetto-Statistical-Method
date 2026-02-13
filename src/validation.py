"""
Script di validazione per la SVD custom.

Confronta i risultati dell'implementazione SVD "from scratch" con
`numpy.linalg.svd` usando diverse matrici di test per verificarne
la correttezza numerica.
"""

import numpy as np
from src.svd import svd, svd_compact, reconstruct
from src.image_utils import (
    load_image_as_matrix,
    apply_mean_centering,
    remove_mean_centering,
    split_into_blocks,
    merge_blocks,
    save_image,
)


def compare_singular_values(sigma_custom: np.ndarray, sigma_numpy: np.ndarray, label: str = "") -> dict:
    """
    Confronta i valori singolari tra la SVD custom e numpy.

    L'errore relativo viene calcolato solo sui valori singolari significativi
    (sopra una soglia numerica), per evitare confronti fuorvianti tra
    valori quasi-zero in matrici di rango basso.

    Ritorna un dizionario con le metriche di confronto.
    """
    # Allinea le lunghezze
    k = min(len(sigma_custom), len(sigma_numpy))
    s_custom = sigma_custom[:k]
    s_numpy = sigma_numpy[:k]

    abs_error = np.abs(s_custom - s_numpy)

    # Per l'errore relativo, consideriamo solo i valori singolari significativi.
    # Soglia: eps_macchina * max(sigma) * max(dimensioni_matrice)
    sv_threshold = np.finfo(np.float64).eps * max(s_numpy[0] if len(s_numpy) > 0 else 1.0, 1.0) * 100
    significant_mask = s_numpy > sv_threshold

    if np.any(significant_mask):
        rel_error_significant = abs_error[significant_mask] / s_numpy[significant_mask]
        max_rel = float(np.max(rel_error_significant))
        mean_rel = float(np.mean(rel_error_significant))
    else:
        max_rel = 0.0
        mean_rel = 0.0

    n_significant = int(np.sum(significant_mask))
    n_negligible = k - n_significant

    result = {
        "label": label,
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "max_rel_error": max_rel,
        "mean_rel_error": mean_rel,
        "n_significant": n_significant,
        "n_negligible": n_negligible,
        "sigma_custom": s_custom,
        "sigma_numpy": s_numpy,
    }
    return result


def validate_svd_decomposition(X: np.ndarray, U: np.ndarray, sigma: np.ndarray, Vt: np.ndarray, label: str = "") -> dict:
    """
    Valida la correttezza della decomposizione SVD verificando:
    1. Ricostruzione: ||X - U·Σ·Vᵀ|| ≈ 0
    2. Ortonormalità di U: UᵀU ≈ I
    3. Ortonormalità di V: VᵀV ≈ I
    """
    k = len(sigma)
    
    # 1. Errore di ricostruzione
    X_reconstructed = reconstruct(U, sigma, Vt)
    reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro') / max(np.linalg.norm(X, 'fro'), 1e-15)

    # 2. Ortonormalità di U (prime k colonne)
    U_k = U[:, :k]
    ortho_U_error = np.linalg.norm(U_k.T @ U_k - np.eye(k), 'fro')

    # 3. Ortonormalità di V (prime k righe di Vt)
    Vt_k = Vt[:k, :]
    ortho_V_error = np.linalg.norm(Vt_k @ Vt_k.T - np.eye(k), 'fro')

    result = {
        "label": label,
        "reconstruction_rel_error": float(reconstruction_error),
        "orthonormality_U_error": float(ortho_U_error),
        "orthonormality_V_error": float(ortho_V_error),
    }
    return result


def print_validation_report(sv_comparison: dict, decomposition_validation: dict) -> None:
    """Stampa un report leggibile dei risultati della validazione."""
    label = sv_comparison.get("label", "Test")
    
    print(f"\n{'=' * 60}")
    print(f"  VALIDAZIONE SVD: {label}")
    print(f"{'=' * 60}")

    print(f"\n  📊 Confronto Valori Singolari (custom vs numpy):")
    print(f"     Max errore assoluto:  {sv_comparison['max_abs_error']:.2e}")
    print(f"     Mean errore assoluto: {sv_comparison['mean_abs_error']:.2e}")
    print(f"     Max errore relativo:  {sv_comparison['max_rel_error']:.2e}  (su {sv_comparison['n_significant']} SV significativi)")
    print(f"     Mean errore relativo: {sv_comparison['mean_rel_error']:.2e}")
    if sv_comparison['n_negligible'] > 0:
        print(f"     ℹ️  {sv_comparison['n_negligible']} valori singolari trascurabili (≈0) esclusi dall'errore relativo")

    print(f"\n  🔄 Qualità della Decomposizione:")
    print(f"     Errore ricostruzione (relativo): {decomposition_validation['reconstruction_rel_error']:.2e}")
    print(f"     Errore ortonormalità U:          {decomposition_validation['orthonormality_U_error']:.2e}")
    print(f"     Errore ortonormalità V:          {decomposition_validation['orthonormality_V_error']:.2e}")

    # Valutazione complessiva
    tol = 1e-6
    sv_ok = sv_comparison['max_rel_error'] < tol
    recon_ok = decomposition_validation['reconstruction_rel_error'] < tol
    ortho_ok = max(decomposition_validation['orthonormality_U_error'], 
                   decomposition_validation['orthonormality_V_error']) < tol

    if sv_ok and recon_ok and ortho_ok:
        print(f"\n  ✅ RISULTATO: SUPERATO — la SVD custom è corretta!")
    else:
        print(f"\n  ⚠️  RISULTATO: ATTENZIONE — possibili discrepanze")
        if not sv_ok:
            print(f"     ❌ Valori singolari non corrispondono (errore relativo > {tol})")
        if not recon_ok:
            print(f"     ❌ Ricostruzione non accurata")
        if not ortho_ok:
            print(f"     ❌ Ortonormalità non soddisfatta")

    print(f"\n{'─' * 60}")


def run_test(X: np.ndarray, label: str) -> bool:
    """
    Esegue un singolo test di validazione su una matrice.

    Ritorna True se il test passa, False altrimenti.
    """
    # SVD custom
    U_custom, sigma_custom, Vt_custom = svd(X)

    # SVD numpy (reference)
    U_numpy, sigma_numpy, Vt_numpy = np.linalg.svd(X, full_matrices=True)

    # Confronta i valori singolari
    sv_comp = compare_singular_values(sigma_custom, sigma_numpy, label)

    # Valida la decomposizione custom
    decomp_val = validate_svd_decomposition(X, U_custom, sigma_custom, Vt_custom, label)

    # Stampa report
    print_validation_report(sv_comp, decomp_val)

    tol = 1e-6
    passed = (
        sv_comp['max_rel_error'] < tol
        and decomp_val['reconstruction_rel_error'] < tol
        and max(decomp_val['orthonormality_U_error'], decomp_val['orthonormality_V_error']) < tol
    )
    return passed


def run_all_tests() -> None:
    """Esegue tutti i test di validazione."""

    print("\n" + "🧪" * 30)
    print("  SUITE DI VALIDAZIONE SVD — from scratch vs numpy")
    print("🧪" * 30)

    results = []
    rng = np.random.default_rng(seed=0)

    # ─── Test 1: Matrice quadrata piccola ───
    X1 = rng.standard_normal((4, 4))
    results.append(run_test(X1, "Matrice 4×4 (quadrata, random)"))

    # ─── Test 2: Matrice rettangolare (più righe) ───
    X2 = rng.standard_normal((6, 3))
    results.append(run_test(X2, "Matrice 6×3 (rettangolare, m > n)"))

    # ─── Test 3: Matrice rettangolare (più colonne) ───
    X3 = rng.standard_normal((3, 6))
    results.append(run_test(X3, "Matrice 3×6 (rettangolare, m < n)"))

    # ─── Test 4: Matrice di rango basso ───
    A = rng.standard_normal((5, 2))
    B = rng.standard_normal((2, 5))
    X4 = A @ B  # rango ≤ 2
    results.append(run_test(X4, "Matrice 5×5 (rango basso ≤ 2)"))

    # ─── Test 5: Matrice identità ───
    X5 = np.eye(5)
    results.append(run_test(X5, "Matrice 5×5 (identità)"))

    # ─── Test 6: Matrice con valori come un blocco immagine (8×8, valori 0-255) ───
    X6 = rng.integers(0, 256, size=(8, 8)).astype(np.float64)
    results.append(run_test(X6, "Matrice 8×8 (blocco immagine simulato, 0-255)"))

    # ─── Test 7: Matrice più grande ───
    X7 = rng.standard_normal((16, 16))
    results.append(run_test(X7, "Matrice 16×16 (dimensione media)"))

    # ─── Riepilogo ───
    print(f"\n\n{'═' * 60}")
    print(f"  📋 RIEPILOGO FINALE")
    print(f"{'═' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"\n  Test superati: {passed}/{total}")
    if passed == total:
        print(f"  🎉 Tutti i test superati! La SVD custom funziona correttamente.")
    else:
        print(f"  ⚠️  {total - passed} test falliti. Rivedere l'implementazione.")
    print()


def test_mean_centering_impact(image_path: str) -> None:
    """
    Testa l'impatto del mean centering sulla qualità della ricostruzione
    di un'immagine tramite SVD con diversi valori di k (approssimazione di rango k).

    Questo test aiuta a decidere se usare o meno il mean centering
    nella pipeline di steganografia.
    """
    print(f"\n{'═' * 60}")
    print(f"  TEST MEAN CENTERING: {image_path}")
    print(f"{'═' * 60}")

    # Carica l'immagine
    X = load_image_as_matrix(image_path, grayscale=True)
    print(f"\n  Dimensioni immagine: {X.shape}")

    # Valori di k da testare
    k_values = [1, 5, 10, 20, 50, min(X.shape) // 2, min(X.shape)]

    print(f"\n  {'k':>5} | {'PSNR (senza MC)':>16} | {'PSNR (con MC)':>14} | {'Migliore':>10}")
    print(f"  {'─' * 55}")

    for k in k_values:
        if k > min(X.shape):
            continue

        # ─── Senza mean centering ───
        U1, s1, Vt1 = svd_compact(X)
        X_recon_no_mc = reconstruct(U1, s1, Vt1, k)
        psnr_no_mc = _compute_psnr(X, X_recon_no_mc)

        # ─── Con mean centering ───
        X_centered, means = apply_mean_centering(X)
        U2, s2, Vt2 = svd_compact(X_centered)
        X_recon_centered = reconstruct(U2, s2, Vt2, k)
        X_recon_mc = remove_mean_centering(X_recon_centered, means)
        psnr_mc = _compute_psnr(X, X_recon_mc)

        better = "= MC" if psnr_mc > psnr_no_mc else "= No MC" if psnr_no_mc > psnr_mc else "= Pari"
        print(f"  {k:>5} | {psnr_no_mc:>14.2f} dB | {psnr_mc:>12.2f} dB | {better:>10}")


def _compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calcola il Peak Signal-to-Noise Ratio (PSNR) tra due immagini."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)


if __name__ == "__main__":
    run_all_tests()
