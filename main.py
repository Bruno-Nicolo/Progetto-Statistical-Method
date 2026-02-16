"""
Main entry point — SVD Image Steganography.

Fase 1: Implementazione Matematica della SVD "from scratch".
Fase 2: Selezione della ROI con YOLOv8.
Fase 3: Embedding del messaggio nei valori singolari.

Questo script:
1. Esegue la suite di validazione della SVD custom vs numpy
2. Testa il mean centering su un'immagine (se fornita)
3. Dimostra la compressione/ricostruzione di un'immagine con la SVD custom
4. (Fase 2) Rileva oggetti con YOLOv8 e seleziona la ROI per la steganografia
5. (Fase 3) Nasconde un messaggio nell'immagine tramite SVD + QIM

Uso:
    python main.py                              # Solo validazione matrici
    python main.py --image <percorso_immagine>  # Validazione + test su immagine
    python main.py --image img.png --yolo       # Fase 2: Rilevamento YOLO + ROI
    python main.py --image foto.png --embed --message "Ciao!"  # Fase 3: Embedding
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


# ═══════════════════════════════════════════════════════════════
#  FASE 1 — Demo SVD su immagine
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  FASE 2 — Rilevamento YOLO e selezione ROI
# ═══════════════════════════════════════════════════════════════

def demo_yolo_roi(
    image_path: str,
    strategy: str = "C",
    box_index: int | None = None,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    output_dir: str = "output",
) -> None:
    """
    Fase 2 — Rileva gli oggetti nell'immagine con YOLOv8 e seleziona la ROI
    per la steganografia.

    Parametri
    ---------
    image_path : str
        Percorso dell'immagine di cover.
    strategy : str
        Strategia di selezione ROI: 'A', 'B', o 'C'.
    box_index : int | None
        Indice del bounding box da usare (solo per strategia A).
    model_name : str
        Nome/percorso del modello YOLOv8 (default: yolov8n.pt).
    confidence : float
        Soglia di confidenza per le detection YOLO.
    output_dir : str
        Cartella di output.
    """
    from src.yolo_roi import (
        load_yolo_model,
        detect_objects,
        select_roi,
        extract_roi_region,
        draw_detections,
        print_detection_report,
    )
    from src.image_utils import load_image_as_matrix, save_image

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  FASE 2 — RILEVAMENTO YOLO + SELEZIONE ROI")
    print(f"{'═' * 60}")
    print(f"\n  📷 Immagine: {image_path}")

    # 1. Carica il modello YOLO
    model = load_yolo_model(model_name)

    # 2. Inferenza: rileva gli oggetti
    print(f"\n  🔍 Esecuzione inferenza (soglia confidenza: {confidence:.0%})...")
    bounding_boxes = detect_objects(model, image_path, confidence_threshold=confidence)

    # 3. Carica l'immagine come matrice per ottenere le dimensioni
    img_matrix = load_image_as_matrix(image_path, grayscale=True)
    image_shape = img_matrix.shape[:2]

    # 4. Selezione della ROI
    roi_result = select_roi(image_shape, bounding_boxes, strategy=strategy, box_index=box_index)

    # 5. Report console
    print_detection_report(bounding_boxes, roi_result, image_shape)

    # 6. Salva immagine annotata con bounding box
    annotated_img = draw_detections(image_path, bounding_boxes, roi_result)
    annotated_path = os.path.join(output_dir, "yolo_detections.png")
    annotated_img.save(annotated_path)
    print(f"\n  💾 Immagine con detection salvata: {annotated_path}")

    # 7. Estrai e salva la ROI
    roi_region = extract_roi_region(img_matrix, roi_result)
    roi_path = os.path.join(output_dir, "roi_extracted.png")
    save_image(roi_region, roi_path)
    print(f"  💾 ROI estratta salvata: {roi_path}")

    # 8. Salva la maschera ROI
    mask_image = (roi_result.mask.astype(np.uint8)) * 255
    mask_path = os.path.join(output_dir, "roi_mask.png")
    save_image(mask_image.astype(np.float64), mask_path)
    print(f"  💾 Maschera ROI salvata: {mask_path}")

    # 9. Demo SVD sulla ROI (solo se strategia A o C con un box selezionato)
    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        roi_matrix = img_matrix[bb.y1:bb.y2, bb.x1:bb.x2].copy()
        h_roi, w_roi = roi_matrix.shape

        print(f"\n  🧮 Demo SVD sulla ROI ({h_roi}×{w_roi}):")

        # SVD compatta sulla ROI
        U_roi, sigma_roi, Vt_roi = svd_compact(roi_matrix)
        print(f"     Rango numerico ROI: {len(sigma_roi)}")

        # Ricostruzione a diversi livelli di k
        k_values = [1, 5, 10, 20]
        k_values = [k for k in k_values if k <= len(sigma_roi)]

        print(f"     {'k':>5} | {'PSNR':>10} | {'File'}")
        print(f"     {'─' * 50}")

        for k in k_values:
            roi_recon = reconstruct(U_roi, sigma_roi, Vt_roi, k)
            psnr = _compute_psnr(roi_matrix, roi_recon)
            filename = f"roi_reconstructed_k{k}.png"
            filepath = os.path.join(output_dir, filename)
            save_image(roi_recon, filepath)
            print(f"     {k:>5} | {psnr:>8.2f} dB | {filepath}")

    print(f"\n  🎉 Fase 2 completata! Output salvati in '{output_dir}/'")


# ═══════════════════════════════════════════════════════════════
#  FASE 3 — Embedding del Messaggio
# ═══════════════════════════════════════════════════════════════

def demo_embed(
    image_path: str,
    message: str,
    strategy: str = "C",
    box_index: int | None = None,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
    output_dir: str = "output",
) -> None:
    """
    Fase 3 — Embedding completo: YOLO ROI + SVD steganography.

    1. Rileva gli oggetti con YOLO e seleziona la ROI
    2. Converte il messaggio in binario
    3. Incorpora il payload nei valori singolari della ROI
    4. Ricostruisce la stego-image
    5. Verifica estraendo il messaggio dalla stego-image

    Parametri
    ---------
    image_path : str
        Percorso dell'immagine di cover.
    message : str
        Il messaggio segreto da nascondere.
    strategy : str
        Strategia di selezione ROI: 'A', 'B', 'C'.
    box_index : int | None
        Indice del bounding box (solo strategia A).
    model_name : str
        Modello YOLOv8.
    confidence : float
        Soglia di confidenza YOLO.
    block_size : int
        Dimensione dei blocchi per la SVD (default: 8).
    sv_range : str
        Quali SV modificare: 'first', 'mid', 'last'.
    delta : float
        Passo di quantizzazione QIM.
    output_dir : str
        Cartella di output.
    """
    from src.yolo_roi import (
        load_yolo_model,
        detect_objects,
        select_roi,
        extract_roi_region,
        draw_detections,
        print_detection_report,
    )
    from src.steganography import (
        text_to_binary,
        binary_to_text,
        embed_in_full_image,
        extract_from_full_image,
        compute_capacity,
        print_embed_report,
    )
    from src.image_utils import load_image_as_matrix, save_image

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  FASE 3 — EMBEDDING DEL MESSAGGIO")
    print(f"{'═' * 60}")
    print(f"\n  📷 Immagine di cover: {image_path}")
    print(f"  💬 Messaggio: \"{message}\"")
    print(f"  📏 Lunghezza messaggio: {len(message)} caratteri ({len(message) * 8 + 8} bit con terminatore)")

    # ─── Step 1: YOLO Detection + ROI ───
    print(f"\n  ── Step 1: Rilevamento YOLO + selezione ROI ──")
    model = load_yolo_model(model_name)
    bounding_boxes = detect_objects(model, image_path, confidence_threshold=confidence)

    img_matrix = load_image_as_matrix(image_path, grayscale=True)
    image_shape = img_matrix.shape[:2]

    roi_result = select_roi(image_shape, bounding_boxes, strategy=strategy, box_index=box_index)
    print_detection_report(bounding_boxes, roi_result, image_shape)

    # Determina le coordinate della ROI
    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        roi_coords = (bb.y1, bb.x1, bb.y2, bb.x2)
    else:
        # Strategia B o nessun oggetto rilevato: usa l'intera immagine
        roi_coords = (0, 0, image_shape[0], image_shape[1])

    y1, x1, y2, x2 = roi_coords
    roi_matrix = img_matrix[y1:y2, x1:x2]
    print(f"\n  📐 ROI selezionata: ({x1},{y1})→({x2},{y2}) = {y2 - y1}×{x2 - x1} pixel")

    # ─── Step 2: Verifica capacità ───
    print(f"\n  ── Step 2: Verifica capacità ──")
    capacity = compute_capacity(roi_matrix, block_size, sv_range)
    payload_bits = text_to_binary(message)

    print(f"     Capacità totale:    {capacity['total_bits']} bit ({capacity['max_text_chars']} caratteri max)")
    print(f"     Payload richiesto:  {len(payload_bits)} bit ({len(message)} caratteri + terminatore)")

    if len(payload_bits) > capacity['total_bits']:
        max_chars = capacity['max_text_chars']
        print(f"\n  ❌ Errore: il messaggio è troppo lungo!")
        print(f"     La ROI può contenere al massimo {max_chars} caratteri.")
        print(f"     Suggerimenti:")
        print(f"       - Riduci il messaggio a {max_chars} caratteri")
        print(f"       - Usa blocchi più piccoli (--block-size 4)")
        print(f"       - Usa sv_range 'first' o 'last' per più bit per blocco")
        print(f"       - Scegli una ROI più grande (strategia B per sfondo)")
        return

    print(f"     ✅ Capacità sufficiente!")

    # ─── Step 3: Embedding ───
    print(f"\n  ── Step 3: Embedding SVD + QIM ──")
    print(f"     Block size:    {block_size}×{block_size}")
    print(f"     SV range:      {sv_range}")
    print(f"     Delta (QIM):   {delta}")

    stego_image, embed_info = embed_in_full_image(
        img_matrix, roi_coords, payload_bits,
        block_size=block_size, sv_range=sv_range, delta=delta,
    )

    print_embed_report(embed_info)

    # ─── Step 4: Salvataggio ───
    print(f"\n  ── Step 4: Salvataggio stego-image ──")
    stego_path = os.path.join(output_dir, "stego_image.png")
    save_image(stego_image, stego_path)
    print(f"     💾 Stego-image salvata: {stego_path}")

    # Salva anche l'immagine annotata con YOLO
    annotated_img = draw_detections(image_path, bounding_boxes, roi_result)
    annotated_path = os.path.join(output_dir, "yolo_detections.png")
    annotated_img.save(annotated_path)
    print(f"     💾 Immagine con detection: {annotated_path}")

    # ─── Step 5: Verifica (estrazione) ───
    print(f"\n  ── Step 5: Verifica — Estrazione del messaggio ──")
    extracted_bits = extract_from_full_image(
        stego_image, roi_coords,
        block_size=block_size, sv_range=sv_range, delta=delta,
        max_bits=len(payload_bits),
    )
    extracted_message = binary_to_text(extracted_bits)

    print(f"     Messaggio originale: \"{message}\"")
    print(f"     Messaggio estratto:  \"{extracted_message}\"")

    if extracted_message == message:
        print(f"\n  ✅ VERIFICA SUPERATA! Il messaggio è stato estratto correttamente.")
    else:
        print(f"\n  ⚠️  VERIFICA FALLITA — il messaggio estratto non corrisponde.")
        # Confronto dettagliato
        matching_bits = np.sum(payload_bits[:len(extracted_bits)] == extracted_bits[:len(payload_bits)])
        total_compare = min(len(payload_bits), len(extracted_bits))
        print(f"     Bit corrispondenti: {matching_bits}/{total_compare} ({matching_bits / total_compare * 100:.1f}%)")

    # ─── Step 6: Metriche visive (anteprima Fase 5) ───
    psnr = _compute_psnr(img_matrix, stego_image)
    print(f"\n  📊 Qualità visiva:")
    print(f"     PSNR (cover vs stego): {psnr:.2f} dB")
    if psnr > 40:
        print(f"     ✅ PSNR > 40 dB → alterazioni impercettibili!")
    elif psnr > 30:
        print(f"     ⚠️  PSNR 30-40 dB → alterazioni minime, possibili lievi artefatti.")
    else:
        print(f"     ❌ PSNR < 30 dB → alterazioni visibili. Prova un delta più basso.")

    print(f"\n  🎉 Fase 3 completata! Output salvati in '{output_dir}/'")


# ═══════════════════════════════════════════════════════════════
#  CLI — Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SVD Image Steganography — Validazione, Demo, YOLO ROI e Embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py                                        # Solo validazione SVD
  python main.py --image foto.png                       # Validazione + demo immagine
  python main.py --image foto.png --no-validate         # Solo demo immagine
  python main.py --image foto.png --mean-centering-test # Test mean centering
  python main.py --image foto.png --yolo                # Fase 2: YOLO detection + ROI
  python main.py --image foto.png --yolo --strategy A   # YOLO con strategia A (soggetti)
  python main.py --image foto.png --yolo --strategy B   # YOLO con strategia B (sfondo)
  python main.py --image foto.png --yolo --strategy C   # YOLO con strategia C (auto, default)
  python main.py --image foto.png --yolo --box-index 1  # Usa il 2° bounding box (strategia A)

  # Fase 3: Embedding del messaggio
  python main.py --image foto.png --embed --message "Messaggio segreto"
  python main.py --image foto.png --embed --message "Test" --sv-range first
  python main.py --image foto.png --embed --message "Test" --sv-range last
  python main.py --image foto.png --embed --message "Test" --block-size 16 --delta 20
        """,
    )
    # ── Fase 1 ──
    parser.add_argument("--image", "-i", type=str, help="Percorso di un'immagine per la demo SVD")
    parser.add_argument("--output", "-o", type=str, default="output", help="Directory di output (default: output)")
    parser.add_argument("--no-validate", action="store_true", help="Salta la validazione matrici")
    parser.add_argument("--mean-centering-test", action="store_true", help="Testa l'impatto del mean centering")

    # ── Fase 2 ──
    parser.add_argument("--yolo", action="store_true", help="Attiva il rilevamento YOLO e la selezione della ROI (Fase 2)")
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="C",
        choices=["A", "B", "C"],
        help=(
            "Strategia di embedding per la selezione della ROI: "
            "A=soggetti (bounding box), B=sfondo, C=auto (bbox più grande). "
            "Default: C"
        ),
    )
    parser.add_argument(
        "--box-index",
        type=int,
        default=None,
        help="Indice del bounding box da usare (solo strategia A). Default: il più grande.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Nome o percorso del modello YOLOv8 (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="Soglia di confidenza per le detection YOLO (default: 0.25)",
    )

    # ── Fase 3 ──
    parser.add_argument("--embed", action="store_true", help="Attiva l'embedding del messaggio (Fase 3)")
    parser.add_argument(
        "--message", "-m",
        type=str,
        default=None,
        help="Il messaggio segreto da nascondere nella stego-image.",
    )
    parser.add_argument(
        "--sv-range",
        type=str,
        default="mid",
        choices=["first", "mid", "last"],
        help=(
            "Quali valori singolari modificare per l'embedding: "
            "first=primi (robusto, artefatti visibili), "
            "mid=intermedi (miglior compromesso, default), "
            "last=ultimi (invisibile, fragile a JPEG)"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Dimensione dei blocchi per la SVD (default: 8). Valori comuni: 4, 8, 16.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=15.0,
        help="Passo di quantizzazione QIM (default: 15.0). Più grande = più robusto ma più visibile.",
    )

    args = parser.parse_args()

    # Step 1: Validazione SVD (Fase 1)
    if not args.no_validate and not args.yolo and not args.embed:
        run_all_tests()

    # Step 2: Demo SVD su immagine (Fase 1)
    if args.image and not args.yolo and not args.embed:
        if not os.path.exists(args.image):
            print(f"\n  ❌ Errore: file '{args.image}' non trovato!")
            sys.exit(1)

        demo_image_svd(args.image, args.output)

        # Step 3: Test mean centering (opzionale)
        if args.mean_centering_test:
            test_mean_centering_impact(args.image)

    # Step 4: Fase 2 — YOLO detection + ROI
    if args.yolo and not args.embed:
        if not args.image:
            print("\n  ❌ Errore: --yolo richiede --image <percorso_immagine>!")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"\n  ❌ Errore: file '{args.image}' non trovato!")
            sys.exit(1)

        demo_yolo_roi(
            image_path=args.image,
            strategy=args.strategy,
            box_index=args.box_index,
            model_name=args.yolo_model,
            confidence=args.yolo_confidence,
            output_dir=args.output,
        )

    # Step 5: Fase 3 — Embedding del messaggio
    if args.embed:
        if not args.image:
            print("\n  ❌ Errore: --embed richiede --image <percorso_immagine>!")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"\n  ❌ Errore: file '{args.image}' non trovato!")
            sys.exit(1)
        if not args.message:
            print("\n  ❌ Errore: --embed richiede --message <messaggio_segreto>!")
            sys.exit(1)

        demo_embed(
            image_path=args.image,
            message=args.message,
            strategy=args.strategy,
            box_index=args.box_index,
            model_name=args.yolo_model,
            confidence=args.yolo_confidence,
            block_size=args.block_size,
            sv_range=args.sv_range,
            delta=args.delta,
            output_dir=args.output,
        )

    if not args.image and args.no_validate and not args.yolo and not args.embed:
        print("  ⚠️  Nessuna operazione eseguita. Usa --help per vedere le opzioni.")


if __name__ == "__main__":
    main()
