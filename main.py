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

IMAGE_PATH = "percorso/alla/tua/immagine.png"

MESSAGE = "Messaggio segreto"

OUTPUT_DIR = "output"

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25
STRATEGY = "C"
BOX_INDEX = None

BLOCK_SIZE = 8
SV_RANGE = "mid"
DELTA = 15.0

RUN_VALIDATION = True
RUN_MEAN_CENTERING_TEST = False
RUN_SVD_DEMO = True
RUN_YOLO = True
RUN_EMBED = True
RUN_EXTRACT = True

def demo_image_svd(image_path: str, output_dir: str = "output") -> None:

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'' * 60}")
    print("DEMO SVD SU IMMAGINE")
    print(f"{'' * 60}")
    print(f"Immagine: {image_path}")

    X = load_image_as_matrix(image_path, grayscale=True)
    h, w = X.shape
    print(f"Dimensioni: {h}×{w}")
    print(f"Rango massimo: {min(h, w)}")

    print("Computing SVD...")
    U, sigma, Vt = svd_compact(X)
    print(f"SVD done. Rank: {len(sigma)}")

    _, sigma_np, _ = np.linalg.svd(X, full_matrices=False)
    k_test = min(10, len(sigma))
    max_sv_error = np.max(np.abs(sigma[:k_test] - sigma_np[:k_test]))
    print(f"Max error: {k_test} vs numpy: {max_sv_error:.2e}")

    k_values = [1, 5, 10, 20, 50, 100]
    k_values = [k for k in k_values if k <= len(sigma)]

    print("Ricostruzioni con diversi valori di k:")
    print(f"{'k':>5}  {'PSNR':>10}  {'File'}")
    print(f"{'' * 50}")

    for k in k_values:
        X_recon = reconstruct(U, sigma, Vt, k)
        psnr = _compute_psnr(X, X_recon)
        filename = f"reconstructed_k{k}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(X_recon, filepath)
        print(f"{k:>5}  {psnr:>8.2f} dB  {filepath}")

    print("Demo SVD per blocchi (8×8):")
    block_size = 8
    blocks, positions, original_shape = split_into_blocks(X, block_size)
    print(f"Numero blocchi: {len(blocks)}")

    reconstructed_blocks = []
    for block in blocks:
        U_b, s_b, Vt_b = svd_compact(block)
        block_recon = reconstruct(U_b, s_b, Vt_b)
        reconstructed_blocks.append(block_recon)

    X_block_recon = merge_blocks(reconstructed_blocks, positions, original_shape, block_size)
    psnr_blocks = _compute_psnr(X, X_block_recon)
    block_filepath = os.path.join(output_dir, "reconstructed_blocks_full.png")
    save_image(X_block_recon, block_filepath)
    print("PSNR ricostruzione blocchi (full rank): {psnr_blocks:.2f} dB")
    print(f"Salvato: {block_filepath}")

    print("Done. Saved in '{output_dir}/'")

def demo_yolo_roi(
    image_path: str,
    strategy: str = "C",
    box_index: int | None = None,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    output_dir: str = "output",
) -> None:

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

    print(f"{'' * 60}")
    print("FASE 2 — RILEVAMENTO YOLO + SELEZIONE ROI")
    print(f"{'' * 60}")
    print(f"Immagine: {image_path}")

    model = load_yolo_model(model_name)

    print("Inference (conf: {confidence:.0%})...")
    bounding_boxes = detect_objects(model, image_path, confidence_threshold=confidence)

    img_matrix = load_image_as_matrix(image_path, grayscale=True)
    image_shape = img_matrix.shape[:2]

    roi_result = select_roi(image_shape, bounding_boxes, strategy=strategy, box_index=box_index)

    print_detection_report(bounding_boxes, roi_result, image_shape)

    annotated_img = draw_detections(image_path, bounding_boxes, roi_result)
    annotated_path = os.path.join(output_dir, "yolo_detections.png")
    annotated_img.save(annotated_path)
    print(f"Detection saved: {annotated_path}")

    roi_region = extract_roi_region(img_matrix, roi_result)
    roi_path = os.path.join(output_dir, "roi_extracted.png")
    save_image(roi_region, roi_path)
    print(f"ROI saved: {roi_path}")

    mask_image = (roi_result.mask.astype(np.uint8)) * 255
    mask_path = os.path.join(output_dir, "roi_mask.png")
    save_image(mask_image.astype(np.float64), mask_path)
    print(f"Mask saved: {mask_path}")

    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        roi_matrix = img_matrix[bb.y1:bb.y2, bb.x1:bb.x2].copy()
        h_roi, w_roi = roi_matrix.shape

        print("Demo SVD sulla ROI ({h_roi}×{w_roi}):")

        U_roi, sigma_roi, Vt_roi = svd_compact(roi_matrix)
        print(f"Rango numerico ROI: {len(sigma_roi)}")

        k_values = [1, 5, 10, 20]
        k_values = [k for k in k_values if k <= len(sigma_roi)]

        print(f"{'k':>5}  {'PSNR':>10}  {'File'}")
        print(f"{'' * 50}")

        for k in k_values:
            roi_recon = reconstruct(U_roi, sigma_roi, Vt_roi, k)
            psnr = _compute_psnr(roi_matrix, roi_recon)
            filename = f"roi_reconstructed_k{k}.png"
            filepath = os.path.join(output_dir, filename)
            save_image(roi_recon, filepath)
            print(f"{k:>5}  {psnr:>8.2f} dB  {filepath}")

    print("Fase 2 completata! Output salvati in '{output_dir}/'")

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

    print(f"{'' * 60}")
    print("FASE 3 — EMBEDDING DEL MESSAGGIO")
    print(f"{'' * 60}")
    print(f"Immagine di cover: {image_path}")
    print("Messaggio: \"{message}\"")
    print("Lunghezza messaggio: {len(message)} caratteri ({len(message) * 7 + 7} bit con terminatore)")

    print("Step 1: YOLO ROI ")
    model = load_yolo_model(model_name)
    bounding_boxes = detect_objects(model, image_path, confidence_threshold=confidence)

    img_matrix = load_image_as_matrix(image_path, grayscale=True)
    image_shape = img_matrix.shape[:2]

    roi_result = select_roi(image_shape, bounding_boxes, strategy=strategy, box_index=box_index)
    print_detection_report(bounding_boxes, roi_result, image_shape)

    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        roi_coords = (bb.y1, bb.x1, bb.y2, bb.x2)
    else:

        roi_coords = (0, 0, image_shape[0], image_shape[1])

    y1, x1, y2, x2 = roi_coords
    roi_matrix = img_matrix[y1:y2, x1:x2]
    print("ROI selezionata: ({x1},{y1})→({x2},{y2}) = {y2  y1}×{x2  x1} pixel")

    print("Step 2: Capacity check ")
    capacity = compute_capacity(roi_matrix, block_size, sv_range)
    payload_bits = text_to_binary(message)

    print("Capacità totale:    {capacity['total_bits']} bit ({capacity['max_text_chars']} caratteri max)")
    print("Payload richiesto:  {len(payload_bits)} bit ({len(message)} caratteri + terminatore)")

    if len(payload_bits) > capacity['total_bits']:
        max_chars = capacity['max_text_chars']
        print("Errore: il messaggio è troppo lungo!")
        print("La ROI può contenere al massimo {max_chars} caratteri.")
        print("Suggerimenti:")
        print("- Riduci il messaggio a {max_chars} caratteri")
        print("- Usa blocchi più piccoli (block_size = 4)")
        print("- Usa sv_range 'first' o 'last' per più bit per blocco")
        print("- Scegli una ROI più grande (strategia B per sfondo)")
        return

    print("Capacità sufficiente!")

    print("Step 3: SVD Embedding ")
    print(f"Block size:    {block_size}×{block_size}")
    print(f"SV range:      {sv_range}")
    print(f"Delta:         {delta}")

    stego_image, embed_info = embed_in_full_image(
        img_matrix, roi_coords, payload_bits,
        block_size=block_size, sv_range=sv_range, delta=delta,
    )

    print_embed_report(embed_info)

    print("Step 4: Salvataggio stegoimage ")
    stego_path = os.path.join(output_dir, "stego_image.png")
    save_image(stego_image, stego_path)
    print(f"Stego-image salvata: {stego_path}")

    annotated_img = draw_detections(image_path, bounding_boxes, roi_result)
    annotated_path = os.path.join(output_dir, "yolo_detections.png")
    annotated_img.save(annotated_path)
    print(f"Immagine con detection: {annotated_path}")

    print("Step 5: Extraction verification ")
    extracted_bits = extract_from_full_image(
        stego_image, img_matrix, roi_coords,
        block_size=block_size, sv_range=sv_range,
        max_bits=len(payload_bits),
    )
    extracted_message = binary_to_text(extracted_bits)

    print("Messaggio originale: \"{message}\"")
    print("Messaggio estratto:  \"{extracted_message}\"")

    if extracted_message == message:
        print("Verification passed.")
    else:
        print("VERIFICA FALLITA — il messaggio estratto non corrisponde.")

        matching_bits = np.sum(payload_bits[:len(extracted_bits)] == extracted_bits[:len(payload_bits)])
        total_compare = min(len(payload_bits), len(extracted_bits))
        print("Bit corrispondenti: {matching_bits}/{total_compare} ({matching_bits / total_compare * 100:.1f}%)")

    psnr = _compute_psnr(img_matrix, stego_image)
    print("Qualità visiva:")
    print("PSNR (cover vs stego): {psnr:.2f} dB")
    if psnr > 40:
        print("PSNR > 40 dB → alterazioni impercettibili!")
    elif psnr > 30:
        print("PSNR 30-40 dB → alterazioni minime, possibili lievi artefatti.")
    else:
        print("PSNR < 30 dB → alterazioni visibili. Prova un delta più basso.")

    print("Fase 3 completata! Output salvati in '{output_dir}/'")

def demo_extract(
    stego_image_path: str,
    original_image_path: str,
    strategy: str = "C",
    box_index: int | None = None,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
    output_dir: str = "output",
) -> None:

    from src.yolo_roi import (
        load_yolo_model,
        detect_objects,
        select_roi,
        draw_detections,
        print_detection_report,
    )
    from src.steganography import (
        binary_to_text,
        extract_from_full_image,
        compute_capacity,
    )
    from src.image_utils import load_image_as_matrix, save_image

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'' * 60}")
    print("FASE 4 — ESTRAZIONE DEL MESSAGGIO (INFORMED)")
    print(f"{'' * 60}")
    print(f"Stegoimage: {stego_image_path}")
    print(f"Immagine originale: {original_image_path}")

    print("Step 1: Rilevamento YOLO — individuare la ROI ")
    print("(YOLO deve ritrovare gli stessi oggetti rilevati in fase di embedding)")

    model = load_yolo_model(model_name)
    bounding_boxes = detect_objects(model, stego_image_path, confidence_threshold=confidence)

    stego_matrix = load_image_as_matrix(stego_image_path, grayscale=True)
    original_matrix = load_image_as_matrix(original_image_path, grayscale=True)
    image_shape = stego_matrix.shape[:2]

    roi_result = select_roi(image_shape, bounding_boxes, strategy=strategy, box_index=box_index)
    print_detection_report(bounding_boxes, roi_result, image_shape)

    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        roi_coords = (bb.y1, bb.x1, bb.y2, bb.x2)
    else:

        roi_coords = (0, 0, image_shape[0], image_shape[1])

    y1, x1, y2, x2 = roi_coords
    roi_matrix = stego_matrix[y1:y2, x1:x2]
    print("ROI individuata: ({x1},{y1})→({x2},{y2}) = {y2  y1}×{x2  x1} pixel")

    print("Step 2: Informazioni ROI e parametri di estrazione ")
    capacity = compute_capacity(roi_matrix, block_size, sv_range)
    print(f"Block size:          {block_size}×{block_size}")
    print(f"SV range:            {sv_range}")
    print(f"Delta:               {delta}")
    print(f"Blocchi nella ROI:   {capacity['n_blocks']}")
    print("Capacità massima:    {capacity['total_bits']} bit ({capacity['max_text_chars']} caratteri)")

    print("Step 3: Estrazione informed (SVD + confronto con originale) ")
    print("Decomposizione SVD dei blocchi ed estrazione dei bit...")

    extracted_bits = extract_from_full_image(
        stego_matrix, original_matrix, roi_coords,
        block_size=block_size, sv_range=sv_range,
    )

    print(f"Bit estratti: {len(extracted_bits)}")

    print("Step 4: Decodifica del payload (ASCII 7bit) ")
    extracted_message = binary_to_text(extracted_bits)

    print(f"{'' * 56}")
    print("Extracted:")
    print(f"{'' * 56}")
    if extracted_message:
        print("\"{extracted_message}\"")
        print(f"{'' * 56}")
        print("Lunghezza: {len(extracted_message)} caratteri")
    else:
        print("(nessun messaggio rilevato o messaggio vuoto)")
        print(f"{'' * 56}")
        print("Possibili cause:")
        print("- Parametri non corrispondenti (block-size, sv-range, delta)")
        print("- Strategia ROI diversa da quella usata nell'embedding")
        print("- L'immagine è stata compressa o alterata dopo l'embedding")
        print("- L'immagine originale fornita non è corretta")
    print(f"{'' * 56}")

    annotated_img = draw_detections(stego_image_path, bounding_boxes, roi_result)
    annotated_path = os.path.join(output_dir, "extract_yolo_detections.png")
    annotated_img.save(annotated_path)
    print(f"Detection YOLO sull'immagine stego: {annotated_path}")

    print("Fase 4 completata!")

if __name__ == "__main__":

    if RUN_VALIDATION:
        run_all_tests()

    needs_image = RUN_SVD_DEMO or RUN_MEAN_CENTERING_TEST or RUN_YOLO or RUN_EMBED or RUN_EXTRACT
    if needs_image:
        if not os.path.exists(IMAGE_PATH):
            print("Errore: file '{IMAGE_PATH}' non trovato!")
            print("Configura IMAGE_PATH nella sezione CONFIGURAZIONE di main.py")
            sys.exit(1)

    if RUN_SVD_DEMO:
        demo_image_svd(IMAGE_PATH, OUTPUT_DIR)

    if RUN_MEAN_CENTERING_TEST:
        test_mean_centering_impact(IMAGE_PATH)

    if RUN_YOLO:
        demo_yolo_roi(
            image_path=IMAGE_PATH,
            strategy=STRATEGY,
            box_index=BOX_INDEX,
            model_name=YOLO_MODEL,
            confidence=YOLO_CONFIDENCE,
            output_dir=OUTPUT_DIR,
        )

    if RUN_EMBED:
        demo_embed(
            image_path=IMAGE_PATH,
            message=MESSAGE,
            strategy=STRATEGY,
            box_index=BOX_INDEX,
            model_name=YOLO_MODEL,
            confidence=YOLO_CONFIDENCE,
            block_size=BLOCK_SIZE,
            sv_range=SV_RANGE,
            delta=DELTA,
            output_dir=OUTPUT_DIR,
        )

    if RUN_EXTRACT:
        stego_path = os.path.join(OUTPUT_DIR, "stego_image.png")
        if not os.path.exists(stego_path):
            print("Errore: stegoimage '{stego_path}' non trovata!")
            print("Esegui prima l'embedding (RUN_EMBED = True).")
            sys.exit(1)

        demo_extract(
            stego_image_path=stego_path,
            original_image_path=IMAGE_PATH,
            strategy=STRATEGY,
            box_index=BOX_INDEX,
            model_name=YOLO_MODEL,
            confidence=YOLO_CONFIDENCE,
            block_size=BLOCK_SIZE,
            sv_range=SV_RANGE,
            delta=DELTA,
            output_dir=OUTPUT_DIR,
        )

    if not any([RUN_VALIDATION, RUN_SVD_DEMO, RUN_MEAN_CENTERING_TEST,
                RUN_YOLO, RUN_EMBED, RUN_EXTRACT]):
        print("Nessuna operazione eseguita. Abilita almeno una fase nella CONFIGURAZIONE.")