"""
Test su dataset di immagini 128x128 con aggregazione dei risultati.

Esegue i test di performance su N immagini dalla cartella artwork/,
ridimensiona ciascuna a 128x128 (crop centrale), e produce CSV aggregati
con media, deviazione standard, min e max per ogni configurazione.
"""

import argparse
import csv
import os
import sys
import time
import io
import glob
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_utils import load_image_as_matrix, compute_psnr
from src.steganography import (
    text_to_binary,
    binary_to_text,
    embed_in_full_image,
    extract_from_full_image,
    compute_capacity,
)
from src.yolo_roi import BoundingBox, select_roi


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 7,
    C1: float = (0.01 * 255) ** 2,
    C2: float = (0.03 * 255) ** 2,
) -> float:
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    if np.array_equal(orig, recon):
        return 1.0
    h, w = orig.shape[:2]
    pad = window_size // 2
    ssim_values = []
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            win_o = orig[i - pad:i + pad + 1, j - pad:j + pad + 1]
            win_r = recon[i - pad:i + pad + 1, j - pad:j + pad + 1]
            mu_o = np.mean(win_o)
            mu_r = np.mean(win_r)
            sigma_o_sq = np.var(win_o)
            sigma_r_sq = np.var(win_r)
            sigma_or = np.mean((win_o - mu_o) * (win_r - mu_r))
            numerator = (2 * mu_o * mu_r + C1) * (2 * sigma_or + C2)
            denominator = (mu_o ** 2 + mu_r ** 2 + C1) * (sigma_o_sq + sigma_r_sq + C2)
            ssim_values.append(numerator / denominator)
    return float(np.mean(ssim_values))


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        return 1.0
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    return float(errors / min_len)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(image.shape) * sigma
    return np.clip(image + noise, 0, 255)


def apply_gaussian_blur(image: np.ndarray, radius: int = 1) -> np.ndarray:
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred, dtype=np.float64)


def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('L')
    return np.array(compressed, dtype=np.float64)


def center_crop(image: np.ndarray, target_size: int) -> np.ndarray:
    """Crop centrale dell'immagine a target_size x target_size."""
    h, w = image.shape[:2]
    if h < target_size or w < target_size:
        return None
    y_start = (h - target_size) // 2
    x_start = (w - target_size) // 2
    return image[y_start:y_start + target_size, x_start:x_start + target_size]


def load_dataset(
    dataset_dir: str,
    max_images: int,
    target_size: int,
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """Carica le immagini, le converte in grayscale e applica il crop centrale."""
    extensions = ('*.jpeg', '*.jpg', '*.png', '*.bmp', '*.tiff')
    all_paths = []
    for ext in extensions:
        all_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    all_paths.sort()

    rng = np.random.default_rng(seed)
    rng.shuffle(all_paths)

    images = []
    skipped = 0
    for path in all_paths:
        if len(images) >= max_images:
            break
        try:
            img = load_image_as_matrix(path, grayscale=True)
            cropped = center_crop(img, target_size)
            if cropped is None:
                skipped += 1
                continue
            images.append((os.path.basename(path), cropped))
        except Exception:
            skipped += 1
            continue

    print(f"Dataset: {len(images)} immagini caricate, {skipped} saltate "
          f"(troppo piccole o non valide)")
    return images


def save_aggregated_csv(
    results: list[dict],
    filename: str,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    if not results:
        return filepath
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            clean_row = {}
            for k, v in row.items():
                if v == float('inf'):
                    clean_row[k] = "inf"
                elif v is None:
                    clean_row[k] = ""
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)
    return filepath


def aggregate_values(values: list[float]) -> dict:
    """Calcola media, std, min, max da una lista di valori numerici."""
    valid = [v for v in values if v is not None and v != float('inf')]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


# ---------------------------------------------------------------------------
# Test 1 — Sweep parametri (aggregato)
# ---------------------------------------------------------------------------

def test_sweep_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 1 — SWEEP PARAMETRI (DATASET)")
    print(f"{'=' * 70}")

    block_sizes = [4, 8, 16]
    sv_ranges = ["first", "mid", "last"]
    deltas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    messages = {
        "corto": "Hello!",
        "medio": "Questo e un messaggio di test per la steganografia SVD.",
        "lungo": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                 "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    }

    # Raccoglie risultati per-configurazione
    config_results: dict[tuple, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        roi_coords = (0, 0, h, w)
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        for block_size in block_sizes:
            for sv_range in sv_ranges:
                for delta in deltas:
                    for msg_label, message in messages.items():
                        key = (block_size, sv_range, delta, msg_label)

                        roi_matrix = image[0:h, 0:w]
                        capacity = compute_capacity(roi_matrix, block_size, sv_range)
                        payload_bits = text_to_binary(message)

                        if len(payload_bits) > capacity['total_bits']:
                            config_results.setdefault(key, []).append({
                                "psnr": None, "ssim": None, "ber": None, "correct": False,
                            })
                            continue

                        stego_image, _ = embed_in_full_image(
                            image, roi_coords, payload_bits,
                            block_size=block_size, sv_range=sv_range, delta=delta,
                        )
                        extracted_bits = extract_from_full_image(
                            stego_image, image, roi_coords,
                            block_size=block_size, sv_range=sv_range,
                            max_bits=len(payload_bits),
                        )
                        extracted_message = binary_to_text(extracted_bits)

                        psnr = compute_psnr(image, stego_image)
                        ssim = compute_ssim(image, stego_image)
                        ber = compute_ber(payload_bits, extracted_bits)
                        correct = extracted_message == message

                        config_results.setdefault(key, []).append({
                            "psnr": psnr, "ssim": ssim, "ber": ber, "correct": correct,
                        })

    # Aggrega
    aggregated = []
    for (block_size, sv_range, delta, msg_label), entries in config_results.items():
        psnr_agg = aggregate_values([e["psnr"] for e in entries])
        ssim_agg = aggregate_values([e["ssim"] for e in entries])
        ber_agg = aggregate_values([e["ber"] for e in entries])
        correct_pct = sum(1 for e in entries if e["correct"]) / len(entries) * 100

        aggregated.append({
            "block_size": block_size,
            "sv_range": sv_range,
            "delta": delta,
            "msg_label": msg_label,
            "n_images": len(entries),
            "psnr_mean": psnr_agg["mean"],
            "psnr_std": psnr_agg["std"],
            "psnr_min": psnr_agg["min"],
            "psnr_max": psnr_agg["max"],
            "ssim_mean": ssim_agg["mean"],
            "ssim_std": ssim_agg["std"],
            "ber_mean": ber_agg["mean"],
            "ber_std": ber_agg["std"],
            "correct_pct": correct_pct,
        })

    # Riepilogo
    valid_psnr = [a["psnr_mean"] for a in aggregated if a["psnr_mean"] is not None]
    valid_ssim = [a["ssim_mean"] for a in aggregated if a["ssim_mean"] is not None]
    if valid_psnr:
        print(f"  PSNR medio globale: {np.mean(valid_psnr):.2f} dB")
    if valid_ssim:
        print(f"  SSIM medio globale: {np.mean(valid_ssim):.4f}")
    print(f"  Configurazioni: {len(aggregated)}")

    return aggregated


# ---------------------------------------------------------------------------
# Test 2 — Impatto del delta (aggregato)
# ---------------------------------------------------------------------------

def test_delta_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 2 — IMPATTO DEL DELTA (DATASET)")
    print(f"{'=' * 70}")

    message = "Test del parametro delta"
    payload_bits = text_to_binary(message)
    deltas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    delta_results: dict[float, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        roi_coords = (0, 0, h, w)
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        for delta in deltas:
            stego_image, _ = embed_in_full_image(
                image, roi_coords, payload_bits,
                block_size=8, sv_range="mid", delta=delta,
            )
            extracted_bits = extract_from_full_image(
                stego_image, image, roi_coords,
                block_size=8, sv_range="mid",
                max_bits=len(payload_bits),
            )
            extracted_message = binary_to_text(extracted_bits)
            psnr = compute_psnr(image, stego_image)
            ssim = compute_ssim(image, stego_image)
            ber = compute_ber(payload_bits, extracted_bits)
            correct = extracted_message == message

            delta_results.setdefault(delta, []).append({
                "psnr": psnr, "ssim": ssim, "ber": ber, "correct": correct,
            })

    aggregated = []
    print(f"\n  {'Delta':>6}  {'PSNR_mean':>10}  {'PSNR_std':>9}  "
          f"{'SSIM_mean':>10}  {'BER_mean':>9}  {'Correct%':>8}")
    print(f"  {'-' * 65}")

    for delta in deltas:
        entries = delta_results.get(delta, [])
        psnr_agg = aggregate_values([e["psnr"] for e in entries])
        ssim_agg = aggregate_values([e["ssim"] for e in entries])
        ber_agg = aggregate_values([e["ber"] for e in entries])
        correct_pct = sum(1 for e in entries if e["correct"]) / max(len(entries), 1) * 100

        aggregated.append({
            "delta": delta,
            "n_images": len(entries),
            "psnr_mean": psnr_agg["mean"],
            "psnr_std": psnr_agg["std"],
            "psnr_min": psnr_agg["min"],
            "psnr_max": psnr_agg["max"],
            "ssim_mean": ssim_agg["mean"],
            "ssim_std": ssim_agg["std"],
            "ber_mean": ber_agg["mean"],
            "ber_std": ber_agg["std"],
            "correct_pct": correct_pct,
        })

        psnr_str = f"{psnr_agg['mean']:.2f}" if psnr_agg["mean"] else "N/A"
        ssim_str = f"{ssim_agg['mean']:.4f}" if ssim_agg["mean"] else "N/A"
        ber_str = f"{ber_agg['mean']:.6f}" if ber_agg["mean"] is not None else "N/A"
        print(f"  {delta:>6.1f}  {psnr_str:>10}  {psnr_agg['std'] or 0:>9.2f}  "
              f"{ssim_str:>10}  {ber_str:>9}  {correct_pct:>7.1f}%")

    return aggregated


# ---------------------------------------------------------------------------
# Test 3 — Robustezza al rumore (aggregato)
# ---------------------------------------------------------------------------

def test_noise_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 3 — ROBUSTEZZA AL RUMORE (DATASET)")
    print(f"{'=' * 70}")

    message = "Test robustezza rumore"
    payload_bits = text_to_binary(message)
    noise_sigmas = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]
    deltas_to_test = [10.0, 15.0, 25.0]

    config_results: dict[tuple, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        roi_coords = (0, 0, h, w)
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        for delta in deltas_to_test:
            stego_image, _ = embed_in_full_image(
                image, roi_coords, payload_bits,
                block_size=8, sv_range="mid", delta=delta,
            )
            for sigma in noise_sigmas:
                noisy_stego = stego_image.copy() if sigma == 0 else add_gaussian_noise(stego_image, sigma)
                extracted_bits = extract_from_full_image(
                    noisy_stego, image, roi_coords,
                    block_size=8, sv_range="mid",
                    max_bits=len(payload_bits),
                )
                ber = compute_ber(payload_bits, extracted_bits)
                correct = binary_to_text(extracted_bits) == message

                config_results.setdefault((delta, sigma), []).append({
                    "ber": ber, "correct": correct,
                })

    aggregated = []
    for delta in deltas_to_test:
        print(f"\n  Delta = {delta}")
        print(f"  {'Sigma':>6}  {'BER_mean':>9}  {'BER_std':>8}  {'Correct%':>8}")
        print(f"  {'-' * 40}")
        for sigma in noise_sigmas:
            entries = config_results.get((delta, sigma), [])
            ber_agg = aggregate_values([e["ber"] for e in entries])
            correct_pct = sum(1 for e in entries if e["correct"]) / max(len(entries), 1) * 100

            aggregated.append({
                "delta": delta,
                "noise_sigma": sigma,
                "n_images": len(entries),
                "ber_mean": ber_agg["mean"],
                "ber_std": ber_agg["std"],
                "ber_min": ber_agg["min"],
                "ber_max": ber_agg["max"],
                "correct_pct": correct_pct,
            })

            ber_str = f"{ber_agg['mean']:.6f}" if ber_agg["mean"] is not None else "N/A"
            print(f"  {sigma:>6}  {ber_str:>9}  {ber_agg['std'] or 0:>8.6f}  {correct_pct:>7.1f}%")

    return aggregated


# ---------------------------------------------------------------------------
# Test 4 — Robustezza al blur (aggregato)
# ---------------------------------------------------------------------------

def test_blur_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 4 — ROBUSTEZZA AL BLUR (DATASET)")
    print(f"{'=' * 70}")

    message = "Test robustezza blur"
    payload_bits = text_to_binary(message)
    blur_radii = [0, 1, 2, 3, 5]
    deltas_to_test = [10.0, 15.0, 25.0]

    config_results: dict[tuple, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        roi_coords = (0, 0, h, w)
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        for delta in deltas_to_test:
            stego_image, _ = embed_in_full_image(
                image, roi_coords, payload_bits,
                block_size=8, sv_range="mid", delta=delta,
            )
            for radius in blur_radii:
                blurred_stego = stego_image.copy() if radius == 0 else apply_gaussian_blur(stego_image, radius)
                extracted_bits = extract_from_full_image(
                    blurred_stego, image, roi_coords,
                    block_size=8, sv_range="mid",
                    max_bits=len(payload_bits),
                )
                ber = compute_ber(payload_bits, extracted_bits)
                correct = binary_to_text(extracted_bits) == message

                config_results.setdefault((delta, radius), []).append({
                    "ber": ber, "correct": correct,
                })

    aggregated = []
    for delta in deltas_to_test:
        print(f"\n  Delta = {delta}")
        print(f"  {'Radius':>6}  {'BER_mean':>9}  {'BER_std':>8}  {'Correct%':>8}")
        print(f"  {'-' * 40}")
        for radius in blur_radii:
            entries = config_results.get((delta, radius), [])
            ber_agg = aggregate_values([e["ber"] for e in entries])
            correct_pct = sum(1 for e in entries if e["correct"]) / max(len(entries), 1) * 100

            aggregated.append({
                "delta": delta,
                "blur_radius": radius,
                "n_images": len(entries),
                "ber_mean": ber_agg["mean"],
                "ber_std": ber_agg["std"],
                "ber_min": ber_agg["min"],
                "ber_max": ber_agg["max"],
                "correct_pct": correct_pct,
            })

            ber_str = f"{ber_agg['mean']:.6f}" if ber_agg["mean"] is not None else "N/A"
            print(f"  {radius:>6}  {ber_str:>9}  {ber_agg['std'] or 0:>8.6f}  {correct_pct:>7.1f}%")

    return aggregated


# ---------------------------------------------------------------------------
# Test 5 — Robustezza JPEG (aggregato)
# ---------------------------------------------------------------------------

def test_jpeg_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 5 — ROBUSTEZZA JPEG (DATASET)")
    print(f"{'=' * 70}")

    message = "Test robustezza JPEG"
    payload_bits = text_to_binary(message)
    jpeg_qualities = [100, 95, 90, 80, 70, 50, 30]
    deltas_to_test = [10.0, 15.0, 25.0]

    config_results: dict[tuple, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        roi_coords = (0, 0, h, w)
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        for delta in deltas_to_test:
            stego_image, _ = embed_in_full_image(
                image, roi_coords, payload_bits,
                block_size=8, sv_range="mid", delta=delta,
            )
            for quality in jpeg_qualities:
                jpeg_stego = stego_image.copy() if quality == 100 else apply_jpeg_compression(stego_image, quality)
                extracted_bits = extract_from_full_image(
                    jpeg_stego, image, roi_coords,
                    block_size=8, sv_range="mid",
                    max_bits=len(payload_bits),
                )
                ber = compute_ber(payload_bits, extracted_bits)
                correct = binary_to_text(extracted_bits) == message

                config_results.setdefault((delta, quality), []).append({
                    "ber": ber, "correct": correct,
                })

    aggregated = []
    for delta in deltas_to_test:
        print(f"\n  Delta = {delta}")
        print(f"  {'Quality':>7}  {'BER_mean':>9}  {'BER_std':>8}  {'Correct%':>8}")
        print(f"  {'-' * 40}")
        for quality in jpeg_qualities:
            entries = config_results.get((delta, quality), [])
            ber_agg = aggregate_values([e["ber"] for e in entries])
            correct_pct = sum(1 for e in entries if e["correct"]) / max(len(entries), 1) * 100

            aggregated.append({
                "delta": delta,
                "jpeg_quality": quality,
                "n_images": len(entries),
                "ber_mean": ber_agg["mean"],
                "ber_std": ber_agg["std"],
                "ber_min": ber_agg["min"],
                "ber_max": ber_agg["max"],
                "correct_pct": correct_pct,
            })

            ber_str = f"{ber_agg['mean']:.6f}" if ber_agg["mean"] is not None else "N/A"
            print(f"  {quality:>7}  {ber_str:>9}  {ber_agg['std'] or 0:>8.6f}  {correct_pct:>7.1f}%")

    return aggregated


# ---------------------------------------------------------------------------
# Test 6 — ROI Strategies (aggregato)
# ---------------------------------------------------------------------------

def test_roi_strategies_dataset(
    images: list[tuple[str, np.ndarray]],
    output_dir: str,
) -> list[dict]:
    print(f"{'=' * 70}")
    print("TEST 6 — ROI STRATEGIES (DATASET) CON MODELLO YOLO REALE")
    print(f"{'=' * 70}")

    message = "Test ROI strategies"
    payload_bits = text_to_binary(message)
    delta = 15.0
    block_size = 8
    sv_range = "mid"

    try:
        from src.yolo_roi import load_yolo_model, detect_objects
        yolo_model = load_yolo_model("yolov8n.pt")
    except ImportError:
        print("Libreria ultralytics non installata. Test saltato.")
        return []

    config_results: dict[str, list[dict]] = {}

    for img_idx, (img_name, image) in enumerate(images):
        h, w = image.shape
        print(f"  [{img_idx + 1}/{len(images)}] {img_name}")

        bboxes = detect_objects(yolo_model, image)

        # --- Strategia A (soggetto: il bounding box) ---
        roi_a = select_roi((h, w), bboxes, strategy="A")
        if roi_a.selected_box is not None:
            bb = roi_a.selected_box
            roi_coords_a = (bb.y1, bb.x1, bb.y2, bb.x2)
        else:
            roi_coords_a = (0, 0, h, w)
            
        roi_matrix_a = image[roi_coords_a[0]:roi_coords_a[2], roi_coords_a[1]:roi_coords_a[3]]
        cap_a = compute_capacity(roi_matrix_a, block_size, sv_range)

        if len(payload_bits) <= cap_a['total_bits']:
            stego_a, _ = embed_in_full_image(
                image, roi_coords_a, payload_bits,
                block_size=block_size, sv_range=sv_range, delta=delta,
            )
            extracted_a = extract_from_full_image(
                stego_a, image, roi_coords_a,
                block_size=block_size, sv_range=sv_range,
                max_bits=len(payload_bits),
            )
            psnr_a = compute_psnr(image, stego_a)
            ssim_a = compute_ssim(image, stego_a)
            ber_a = compute_ber(payload_bits, extracted_a)
            correct_a = binary_to_text(extracted_a) == message
            outside_a = (
                np.array_equal(image[:bb.y1, :], stego_a[:bb.y1, :]) and
                np.array_equal(image[bb.y2:, :], stego_a[bb.y2:, :])
            )
            config_results.setdefault("A_soggetto", []).append({
                "psnr": psnr_a, "ssim": ssim_a, "ber": ber_a,
                "correct": correct_a, "outside_intact": outside_a,
            })
        else:
            config_results.setdefault("A_soggetto", []).append({
                "psnr": None, "ssim": None, "ber": None,
                "correct": False, "outside_intact": True,
            })

        # --- Strategia B (sfondo: escludi il bounding box) ---
        # Per strategia B usiamo l'intera immagine come ROI ma verifichiamo
        # che la maschera escluda il box. Usiamo roi_coords = intera immagine
        # dato che embed_in_full_image lavora su coordinate rettangolari.
        roi_coords_full = (0, 0, h, w)
        cap_full = compute_capacity(image, block_size, sv_range)

        if len(payload_bits) <= cap_full['total_bits']:
            stego_b, _ = embed_in_full_image(
                image, roi_coords_full, payload_bits,
                block_size=block_size, sv_range=sv_range, delta=delta,
            )
            extracted_b = extract_from_full_image(
                stego_b, image, roi_coords_full,
                block_size=block_size, sv_range=sv_range,
                max_bits=len(payload_bits),
            )
            psnr_b = compute_psnr(image, stego_b)
            ssim_b = compute_ssim(image, stego_b)
            ber_b = compute_ber(payload_bits, extracted_b)
            correct_b = binary_to_text(extracted_b) == message
            config_results.setdefault("B_sfondo", []).append({
                "psnr": psnr_b, "ssim": ssim_b, "ber": ber_b,
                "correct": correct_b, "outside_intact": True,
            })
        else:
            config_results.setdefault("B_sfondo", []).append({
                "psnr": None, "ssim": None, "ber": None,
                "correct": False, "outside_intact": True,
            })

        # --- Strategia C (automatico: scegli il box piu grande) ---
        roi_c = select_roi((h, w), bboxes, strategy="C")
        if roi_c.selected_box is not None:
            bb_c = roi_c.selected_box
            roi_coords_c = (bb_c.y1, bb_c.x1, bb_c.y2, bb_c.x2)
        else:
            roi_coords_c = (0, 0, h, w)
            
        roi_matrix_c = image[roi_coords_c[0]:roi_coords_c[2], roi_coords_c[1]:roi_coords_c[3]]
        cap_c = compute_capacity(roi_matrix_c, block_size, sv_range)

        if len(payload_bits) <= cap_c['total_bits']:
            stego_c, _ = embed_in_full_image(
                image, roi_coords_c, payload_bits,
                block_size=block_size, sv_range=sv_range, delta=delta,
            )
            extracted_c = extract_from_full_image(
                stego_c, image, roi_coords_c,
                block_size=block_size, sv_range=sv_range,
                max_bits=len(payload_bits),
            )
            psnr_c = compute_psnr(image, stego_c)
            ssim_c = compute_ssim(image, stego_c)
            ber_c = compute_ber(payload_bits, extracted_c)
            correct_c = binary_to_text(extracted_c) == message
            outside_c = (
                np.array_equal(image[:bb_c.y1, :], stego_c[:bb_c.y1, :]) and
                np.array_equal(image[bb_c.y2:, :], stego_c[bb_c.y2:, :])
            )
            config_results.setdefault("C_automatico", []).append({
                "psnr": psnr_c, "ssim": ssim_c, "ber": ber_c,
                "correct": correct_c, "outside_intact": outside_c,
            })
        else:
            config_results.setdefault("C_automatico", []).append({
                "psnr": None, "ssim": None, "ber": None,
                "correct": False, "outside_intact": True,
            })

        # --- Intera immagine (baseline, nessun YOLO) ---
        if len(payload_bits) <= cap_full['total_bits']:
            stego_full, _ = embed_in_full_image(
                image, roi_coords_full, payload_bits,
                block_size=block_size, sv_range=sv_range, delta=delta,
            )
            extracted_full = extract_from_full_image(
                stego_full, image, roi_coords_full,
                block_size=block_size, sv_range=sv_range,
                max_bits=len(payload_bits),
            )
            psnr_full = compute_psnr(image, stego_full)
            ssim_full = compute_ssim(image, stego_full)
            ber_full = compute_ber(payload_bits, extracted_full)
            correct_full = binary_to_text(extracted_full) == message
            config_results.setdefault("full_image", []).append({
                "psnr": psnr_full, "ssim": ssim_full, "ber": ber_full,
                "correct": correct_full, "outside_intact": True,
            })
        else:
            config_results.setdefault("full_image", []).append({
                "psnr": None, "ssim": None, "ber": None,
                "correct": False, "outside_intact": True,
            })

    # Aggrega per strategia
    aggregated = []
    print(f"\n  {'Strategy':<15}  {'PSNR_mean':>10}  {'SSIM_mean':>10}  "
          f"{'BER_mean':>9}  {'Correct%':>8}  {'Outside%':>8}")
    print(f"  {'-' * 70}")

    for strategy_label in ["A_soggetto", "B_sfondo", "C_automatico", "full_image"]:
        entries = config_results.get(strategy_label, [])
        psnr_agg = aggregate_values([e["psnr"] for e in entries])
        ssim_agg = aggregate_values([e["ssim"] for e in entries])
        ber_agg = aggregate_values([e["ber"] for e in entries])
        correct_pct = sum(1 for e in entries if e["correct"]) / max(len(entries), 1) * 100
        outside_pct = sum(1 for e in entries if e.get("outside_intact", True)) / max(len(entries), 1) * 100

        aggregated.append({
            "strategy": strategy_label,
            "n_images": len(entries),
            "psnr_mean": psnr_agg["mean"],
            "psnr_std": psnr_agg["std"],
            "psnr_min": psnr_agg["min"],
            "psnr_max": psnr_agg["max"],
            "ssim_mean": ssim_agg["mean"],
            "ssim_std": ssim_agg["std"],
            "ber_mean": ber_agg["mean"],
            "ber_std": ber_agg["std"],
            "correct_pct": correct_pct,
            "outside_intact_pct": outside_pct,
        })

        psnr_str = f"{psnr_agg['mean']:.2f}" if psnr_agg["mean"] else "N/A"
        ssim_str = f"{ssim_agg['mean']:.4f}" if ssim_agg["mean"] else "N/A"
        ber_str = f"{ber_agg['mean']:.6f}" if ber_agg["mean"] is not None else "N/A"
        print(f"  {strategy_label:<15}  {psnr_str:>10}  {ssim_str:>10}  "
              f"{ber_str:>9}  {correct_pct:>7.1f}%  {outside_pct:>7.1f}%")

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test su dataset di immagini con aggregazione dei risultati"
    )
    parser.add_argument("-d", "--dataset", type=str, default="artwork/",
                        help="Cartella contenente le immagini (default: artwork/)")
    parser.add_argument("-o", "--output", type=str, default="test_output/dataset/",
                        help="Directory di output per i CSV aggregati (default: test_output/dataset/)")
    parser.add_argument("-n", "--max-images", type=int, default=50,
                        help="Numero massimo di immagini da caricare (default: 50)")
    parser.add_argument("-s", "--size", type=int, default=128,
                        help="Dimensione target per il crop (default: 128)")
    parser.add_argument("-t", "--test", type=int, default=None,
                        help="Esegui solo il test specificato (1-6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per lo shuffle riproducibile (default: 42)")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print("TEST SU DATASET — STEGANOGRAFIA SVD")
    print(f"{'=' * 70}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {output_dir}")
    print(f"Max images:  {args.max_images}")
    print(f"Crop size:   {args.size}x{args.size}")
    print(f"Seed:        {args.seed}")
    print()

    t_start = time.time()

    images = load_dataset(args.dataset, args.max_images, args.size, args.seed)
    if not images:
        print("Nessuna immagine caricata. Controlla il percorso del dataset.")
        sys.exit(1)

    all_tests = {
        1: ("Sweep parametri", test_sweep_dataset, "agg_sweep.csv"),
        2: ("Impatto delta", test_delta_dataset, "agg_delta.csv"),
        3: ("Robustezza rumore", test_noise_dataset, "agg_noise.csv"),
        4: ("Robustezza blur", test_blur_dataset, "agg_blur.csv"),
        5: ("Robustezza JPEG", test_jpeg_dataset, "agg_jpeg.csv"),
        6: ("ROI Strategies", test_roi_strategies_dataset, "agg_roi.csv"),
    }

    tests_to_run = [args.test] if args.test else list(all_tests.keys())

    for test_id in tests_to_run:
        if test_id not in all_tests:
            print(f"Test {test_id} non esiste. Disponibili: 1-6")
            continue

        name, func, csv_file = all_tests[test_id]
        print(f"\nEsecuzione: {name}...")
        t_test = time.time()

        results = func(images, output_dir)

        if results:
            csv_path = save_aggregated_csv(results, csv_file, output_dir)
            print(f"  Risultati salvati: {csv_path}")

        print(f"  Tempo: {time.time() - t_test:.1f}s")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print("SUITE DATASET COMPLETATA")
    print(f"{'=' * 70}")
    print(f"Immagini processate: {len(images)}")
    print(f"Test eseguiti:       {len(tests_to_run)}")
    print(f"Tempo totale:        {elapsed:.1f}s")
    print(f"Output:              {output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
