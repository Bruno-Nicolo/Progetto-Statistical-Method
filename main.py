"""
Main entry point — Fase 1: Implementazione Matematica della SVD "from scratch".

Questo script:
1. Esegue la suite di validazione della SVD custom vs numpy
2. Testa il mean centering su un'immagine (se fornita)
3. Dimostra la compressione/ricostruzione di un'immagine con la SVD custom

Uso:
    python main.py                              # Solo validazione matrici
    python main.py --image <percorso_immagine>  # Validazione + test su immagine
"""

import argparse
import os
import sys
import numpy as np

from src.svd import svd, svd_compact, reconstruct
from src.image_utils import (
    load_image_as_matrix,
    apply_mean_centering,
    remove_mean_centering,
    split_into_blocks,
    merge_blocks,
    save_image,
    matrix_to_image,
)
from src.validation import run_all_tests, test_mean_centering_impact, _compute_psnr


def demo_image_svd(image_path: str, output_dir: str = "output") -> None:
    """
    Dimostra la SVD custom su un'immagine reale:
    - Carica l'immagine
    - Applica SVD (con e senza mean centering)
    - Ricostruisce con diversi valori di k
    - Salva le immagini ricostruite

    Parametri
    ---------
    image_path : str
        Percorso dell'immagine di input.
    output_dir : str
        Cartella dove salvare i risultati.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  DEMO SVD SU IMMAGINE")
    print(f"{'═' * 60}")
    print(f"\n  📷 Immagine: {image_path}")

    # Carica immagine
    X = load_image_as_matrix(image_path, grayscale=True)
    h, w = X.shape
    print(f"  📐 Dimensioni: {h}×{w}")
    print(f"  🔢 Rango massimo: {min(h, w)}")

    # ─── SVD sull'intera immagine (senza mean centering) ───
    print(f"\n  ⏳ Calcolo SVD custom (potrebbe richiedere un po' per immagini grandi)...")
    U, sigma, Vt = svd_compact(X)
    print(f"  ✅ SVD calcolata! Rango numerico: {len(sigma)}")

    # Confronta con numpy
    _, sigma_np, _ = np.linalg.svd(X, full_matrices=False)
    k_test = min(10, len(sigma))
    max_sv_error = np.max(np.abs(sigma[:k_test] - sigma_np[:k_test]))
    print(f"  📊 Max errore sui primi {k_test} valori singolari vs numpy: {max_sv_error:.2e}")

    # Ricostruzione con diversi valori di k
    k_values = [1, 5, 10, 20, 50, 100]
    k_values = [k for k in k_values if k <= len(sigma)]

    print(f"\n  Ricostruzioni con diversi valori di k:")
    print(f"  {'k':>5} | {'PSNR':>10} | {'File'}")
    print(f"  {'─' * 50}")

    for k in k_values:
        X_recon = reconstruct(U, sigma, Vt, k)
        psnr = _compute_psnr(X, X_recon)
        filename = f"reconstructed_k{k}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(X_recon, filepath)
        print(f"  {k:>5} | {psnr:>8.2f} dB | {filepath}")

    # ─── SVD per blocchi (8×8): come sarà usata nella steganografia ───
    print(f"\n  🧩 Demo SVD per blocchi (8×8):")
    block_size = 8
    blocks, positions, original_shape = split_into_blocks(X, block_size)
    print(f"     Numero blocchi: {len(blocks)}")

    # Ricostruisci ogni blocco usando SVD compact
    reconstructed_blocks = []
    for block in blocks:
        U_b, s_b, Vt_b = svd_compact(block)
        block_recon = reconstruct(U_b, s_b, Vt_b)
        reconstructed_blocks.append(block_recon)

    X_block_recon = merge_blocks(reconstructed_blocks, positions, original_shape, block_size)
    psnr_blocks = _compute_psnr(X, X_block_recon)
    block_filepath = os.path.join(output_dir, "reconstructed_blocks_full.png")
    save_image(X_block_recon, block_filepath)
    print(f"     PSNR ricostruzione blocchi (full rank): {psnr_blocks:.2f} dB")
    print(f"     Salvato: {block_filepath}")

    print(f"\n  🎉 Demo completata! Output salvati in '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(
        description="SVD from scratch — Validazione e Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py                              # Solo validazione
  python main.py --image foto.png             # Validazione + demo immagine
  python main.py --image foto.png --no-validate  # Solo demo immagine
  python main.py --image foto.png --mean-centering-test  # Test mean centering
        """,
    )
    parser.add_argument("--image", "-i", type=str, help="Percorso di un'immagine per la demo SVD")
    parser.add_argument("--output", "-o", type=str, default="output", help="Directory di output (default: output)")
    parser.add_argument("--no-validate", action="store_true", help="Salta la validazione matrici")
    parser.add_argument("--mean-centering-test", action="store_true", help="Testa l'impatto del mean centering")

    args = parser.parse_args()

    # Step 1: Validazione
    if not args.no_validate:
        run_all_tests()

    # Step 2: Demo su immagine
    if args.image:
        if not os.path.exists(args.image):
            print(f"\n  ❌ Errore: file '{args.image}' non trovato!")
            sys.exit(1)

        demo_image_svd(args.image, args.output)

        # Step 3: Test mean centering (opzionale)
        if args.mean_centering_test:
            test_mean_centering_impact(args.image)

    if not args.image and args.no_validate:
        print("  ⚠️  Nessuna operazione eseguita. Usa --help per vedere le opzioni.")


if __name__ == "__main__":
    main()
