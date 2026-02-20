import argparse
import csv
import os
import sys
import time
import io
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.svd import svd_compact, reconstruct
from src.image_utils import load_image_as_matrix, save_image
from src.steganography import (
    text_to_binary,
    binary_to_text,
    embed_in_full_image,
    extract_from_full_image,
    compute_capacity,
)
from src.image_utils import compute_psnr
from src.yolo_roi import BoundingBox, select_roi

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

def generate_test_image(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            img[i, j] = ((i + j) / (h + w)) * 180 + 30

    cy, cx = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            img[i, j] += 30 * np.sin(dist / 8)

    img += rng.standard_normal((h, w)) * 5

    img = np.clip(img, 0, 255)
    return img

def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(image.shape) * sigma
    noisy = image + noise
    return np.clip(noisy, 0, 255)

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

def test_parameter_sweep(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    output_dir: str = "test_output",
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 1 — SWEEP PARAMETRI")
    print(f"{'' * 70}")

    block_sizes = [4, 8, 16]
    sv_ranges = ["first", "mid", "last"]
    deltas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    messages = {
        "corto": "Hello!",
        "medio": "Questo e un messaggio di test per la steganografia SVD.",
        "lungo": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    }

    results = []
    total = len(block_sizes) * len(sv_ranges) * len(deltas) * len(messages)
    count = 0

    print(f"{'#':>4}  {'Block':>5}  {'SV Range':>8}  {'Delta':>5}  {'Msg':>6}  "
          f"{'PSNR':>8}  {'SSIM':>6}  {'BER':>8}  {'OK':>3}  {'Tempo':>6}")
    print(f"{'' * 85}")

    for block_size in block_sizes:
        for sv_range in sv_ranges:
            for delta in deltas:
                for msg_label, message in messages.items():
                    count += 1

                    y1, x1, y2, x2 = roi_coords
                    roi_matrix = image[y1:y2, x1:x2]
                    capacity = compute_capacity(roi_matrix, block_size, sv_range)
                    payload_bits = text_to_binary(message)

                    if len(payload_bits) > capacity['total_bits']:
                        results.append({
                            "block_size": block_size,
                            "sv_range": sv_range,
                            "delta": delta,
                            "msg_label": msg_label,
                            "msg_len": len(message),
                            "psnr": None,
                            "ssim": None,
                            "ber": None,
                            "correct": False,
                            "time_s": 0,
                            "note": "Capacità insufficiente",
                        })
                        print(f"{count:>4}  {block_size:>5}  {sv_range:>8}  {delta:>5.0f}  "
                              f"{msg_label:>6}  {'':>8}  {'':>6}  {'':>8}  {'':>3}  {'SKIP':>6}")
                        continue

                    t_start = time.time()
                    stego_image, embed_info = embed_in_full_image(
                        image, roi_coords, payload_bits,
                        block_size=block_size, sv_range=sv_range, delta=delta,
                    )

                    extracted_bits = extract_from_full_image(
                        stego_image, image, roi_coords,
                        block_size=block_size, sv_range=sv_range,
                        max_bits=len(payload_bits),
                    )
                    t_elapsed = time.time() - t_start

                    extracted_message = binary_to_text(extracted_bits)

                    psnr = compute_psnr(image, stego_image)
                    ssim = compute_ssim(image, stego_image)
                    ber = compute_ber(payload_bits, extracted_bits)
                    correct = extracted_message == message

                    result = {
                        "block_size": block_size,
                        "sv_range": sv_range,
                        "delta": delta,
                        "msg_label": msg_label,
                        "msg_len": len(message),
                        "psnr": psnr,
                        "ssim": ssim,
                        "ber": ber,
                        "correct": correct,
                        "time_s": t_elapsed,
                        "note": "",
                    }
                    results.append(result)

                    ok_str = "" if correct else ""
                    print(f"{count:>4}  {block_size:>5}  {sv_range:>8}  {delta:>5.0f}  "
                          f"{msg_label:>6}  {psnr:>7.2f}dB  {ssim:>5.4f}  {ber:>8.6f}  "
                          f"{ok_str:>3}  {t_elapsed:>5.1f}s")

    valid = [r for r in results if r["psnr"] is not None]
    correct_count = sum(1 for r in valid if r["correct"])
    skipped = sum(1 for r in results if r["psnr"] is None)
    print(f"Riepilogo: {correct_count}/{len(valid)} test superati, {skipped} saltati (capacità insufficiente)")

    if valid:
        avg_psnr = np.mean([r["psnr"] for r in valid])
        avg_ssim = np.mean([r["ssim"] for r in valid])
        avg_ber = np.mean([r["ber"] for r in valid])
        print(f"PSNR medio: {avg_psnr:.2f} dB  SSIM medio: {avg_ssim:.4f}  BER medio: {avg_ber:.6f}")

    return results

def test_delta_impact(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 2 — IMPATTO DEL DELTA SULLA QUALITÀ VISIVA")
    print(f"{'' * 70}")

    message = "Test del parametro delta"
    payload_bits = text_to_binary(message)
    deltas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    results = []

    print(f"Messaggio: \"{message}\" ({len(message)} caratteri)")
    print("Block size: 8  SV range: mid")
    print(f"{'Delta':>6}  {'PSNR':>8}  {'SSIM':>7}  {'BER':>8}  {'OK':>3}  {'Estratto'}")
    print(f"{'' * 75}")

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

        result = {"delta": delta, "psnr": psnr, "ssim": ssim, "ber": ber, "correct": correct}
        results.append(result)

        ok_str = "" if correct else ""
        display_msg = extracted_message[:40] + "..." if len(extracted_message) > 40 else extracted_message
        print(f"{delta:>6.1f}  {psnr:>7.2f}dB  {ssim:>6.4f}  {ber:>8.6f}  {ok_str:>3}  \"{display_msg}\"")

    return results

def test_capacity_analysis(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 3 — ANALISI DELLA CAPACITÀ")
    print(f"{'' * 70}")

    y1, x1, y2, x2 = roi_coords
    roi_matrix = image[y1:y2, x1:x2]
    h_roi, w_roi = roi_matrix.shape

    print(f"ROI: {h_roi}×{w_roi} pixel ({h_roi * w_roi:,} totali)")

    block_sizes = [4, 8, 16]
    sv_ranges = ["first", "mid", "last"]

    results = []

    print(f"{'Block':>5}  {'SV Range':>8}  {'Blocchi':>7}  {'Bit/Blocco':>10}  "
          f"{'Bit Totali':>10}  {'Max Char':>8}  {'Test':>5}")
    print(f"{'' * 70}")

    for block_size in block_sizes:
        for sv_range in sv_ranges:
            capacity = compute_capacity(roi_matrix, block_size, sv_range)

            max_chars = capacity['max_text_chars']
            if max_chars > 0:
                test_msg = "A" * min(max_chars, 200)
                payload = text_to_binary(test_msg)

                stego_image, _ = embed_in_full_image(
                    image, roi_coords, payload,
                    block_size=block_size, sv_range=sv_range, delta=15.0,
                )
                extracted = extract_from_full_image(
                    stego_image, image, roi_coords,
                    block_size=block_size, sv_range=sv_range,
                    max_bits=len(payload),
                )
                extracted_msg = binary_to_text(extracted)
                correct = extracted_msg == test_msg
                test_str = "" if correct else ""
            else:
                test_str = "N/A"
                correct = False

            result = {
                "block_size": block_size,
                "sv_range": sv_range,
                "n_blocks": capacity['n_blocks'],
                "bits_per_block": capacity['bits_per_block'],
                "total_bits": capacity['total_bits'],
                "max_chars": max_chars,
                "test_passed": correct,
            }
            results.append(result)

            print(f"{block_size:>5}  {sv_range:>8}  {capacity['n_blocks']:>7}  "
                  f"{capacity['bits_per_block']:>10}  {capacity['total_bits']:>10}  "
                  f"{max_chars:>8}  {test_str:>5}")

    return results

def test_robustness_noise(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 4 — ROBUSTEZZA AL RUMORE GAUSSIANO")
    print(f"{'' * 70}")

    message = "Test robustezza rumore"
    payload_bits = text_to_binary(message)
    noise_sigmas = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]

    deltas_to_test = [10.0, 15.0, 25.0]

    results = []

    for delta in deltas_to_test:
        print(f"Delta = {delta} ")
        print(f"{'Sigma':>6}  {'PSNR_noise':>10}  {'BER':>8}  {'OK':>3}  {'Estratto'}")
        print(f"{'' * 65}")

        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=delta,
        )

        for sigma in noise_sigmas:
            if sigma == 0:
                noisy_stego = stego_image.copy()
            else:
                noisy_stego = add_gaussian_noise(stego_image, sigma)

            psnr_noise = compute_psnr(stego_image, noisy_stego) if sigma > 0 else float('inf')

            extracted_bits = extract_from_full_image(
                noisy_stego, image, roi_coords,
                block_size=8, sv_range="mid",
                max_bits=len(payload_bits),
            )
            extracted_message = binary_to_text(extracted_bits)
            ber = compute_ber(payload_bits, extracted_bits)
            correct = extracted_message == message

            result = {
                "delta": delta,
                "noise_sigma": sigma,
                "psnr_noise": psnr_noise,
                "ber": ber,
                "correct": correct,
            }
            results.append(result)

            ok_str = "" if correct else ""
            psnr_str = f"{psnr_noise:.2f}dB" if psnr_noise != float('inf') else "∞"
            display_msg = extracted_message[:30] + "..." if len(extracted_message) > 30 else extracted_message
            print(f"{sigma:>6}  {psnr_str:>10}  {ber:>8.6f}  {ok_str:>3}  \"{display_msg}\"")

    return results

def test_robustness_blur(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 5 — ROBUSTEZZA AL BLUR GAUSSIANO")
    print(f"{'' * 70}")

    message = "Test robustezza blur"
    payload_bits = text_to_binary(message)
    blur_radii = [0, 1, 2, 3, 5]

    deltas_to_test = [10.0, 15.0, 25.0]
    results = []

    for delta in deltas_to_test:
        print(f"Delta = {delta} ")
        print(f"{'Radius':>6}  {'PSNR_blur':>10}  {'BER':>8}  {'OK':>3}  {'Estratto'}")
        print(f"{'' * 65}")

        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=delta,
        )

        for radius in blur_radii:
            if radius == 0:
                blurred_stego = stego_image.copy()
            else:
                blurred_stego = apply_gaussian_blur(stego_image, radius)

            psnr_blur = compute_psnr(stego_image, blurred_stego) if radius > 0 else float('inf')

            extracted_bits = extract_from_full_image(
                blurred_stego, image, roi_coords,
                block_size=8, sv_range="mid",
                max_bits=len(payload_bits),
            )
            extracted_message = binary_to_text(extracted_bits)
            ber = compute_ber(payload_bits, extracted_bits)
            correct = extracted_message == message

            result = {
                "delta": delta,
                "blur_radius": radius,
                "psnr_blur": psnr_blur,
                "ber": ber,
                "correct": correct,
            }
            results.append(result)

            ok_str = "" if correct else ""
            psnr_str = f"{psnr_blur:.2f}dB" if psnr_blur != float('inf') else "∞"
            display_msg = extracted_message[:30] + "..." if len(extracted_message) > 30 else extracted_message
            print(f"{radius:>6}  {psnr_str:>10}  {ber:>8.6f}  {ok_str:>3}  \"{display_msg}\"")

    return results

def test_robustness_jpeg(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 6 — ROBUSTEZZA ALLA COMPRESSIONE JPEG")
    print(f"{'' * 70}")

    message = "Test robustezza JPEG"
    payload_bits = text_to_binary(message)
    jpeg_qualities = [100, 95, 90, 80, 70, 50, 30]

    deltas_to_test = [10.0, 15.0, 25.0]
    results = []

    for delta in deltas_to_test:
        print(f"Delta = {delta} ")
        print(f"{'Quality':>7}  {'PSNR_jpeg':>10}  {'BER':>8}  {'OK':>3}  {'Estratto'}")
        print(f"{'' * 65}")

        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=delta,
        )

        for quality in jpeg_qualities:
            if quality == 100:
                jpeg_stego = stego_image.copy()
            else:
                jpeg_stego = apply_jpeg_compression(stego_image, quality)

            psnr_jpeg = compute_psnr(stego_image, jpeg_stego) if quality < 100 else float('inf')

            extracted_bits = extract_from_full_image(
                jpeg_stego, image, roi_coords,
                block_size=8, sv_range="mid",
                max_bits=len(payload_bits),
            )
            extracted_message = binary_to_text(extracted_bits)
            ber = compute_ber(payload_bits, extracted_bits)
            correct = extracted_message == message

            result = {
                "delta": delta,
                "jpeg_quality": quality,
                "psnr_jpeg": psnr_jpeg,
                "ber": ber,
                "correct": correct,
            }
            results.append(result)

            ok_str = "" if correct else ""
            psnr_str = f"{psnr_jpeg:.2f}dB" if psnr_jpeg != float('inf') else "∞"
            display_msg = extracted_message[:30] + "..." if len(extracted_message) > 30 else extracted_message
            print(f"{quality:>7}  {psnr_str:>10}  {ber:>8.6f}  {ok_str:>3}  \"{display_msg}\"")

    return results

def test_sv_range_visual_impact(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    output_dir: str = "test_output",
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 7 — IMPATTO VISIVO DEI SV RANGE")
    print(f"{'' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    message = "Messaggio di test per confronto visivo"
    payload_bits = text_to_binary(message)
    delta = 15.0

    sv_ranges = ["first", "mid", "last"]
    results = []

    print(f"Messaggio: \"{message}\"")
    print(f"Delta: {delta}")
    print(f"{'SV Range':>8}  {'PSNR':>8}  {'SSIM':>7}  {'Max Diff':>8}  {'Mean Diff':>9}  {'File'}")
    print(f"{'' * 75}")

    for sv_range in sv_ranges:
        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range=sv_range, delta=delta,
        )

        psnr = compute_psnr(image, stego_image)
        ssim = compute_ssim(image, stego_image)

        diff = np.abs(image - stego_image)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        stego_path = os.path.join(output_dir, f"stego_{sv_range}.png")
        save_image(stego_image, stego_path)

        diff_amplified = np.clip(diff * 10, 0, 255)
        diff_path = os.path.join(output_dir, f"diff_{sv_range}.png")
        save_image(diff_amplified, diff_path)

        result = {
            "sv_range": sv_range,
            "psnr": psnr,
            "ssim": ssim,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }
        results.append(result)

        print(f"{sv_range:>8}  {psnr:>7.2f}dB  {ssim:>6.4f}  {max_diff:>8.2f}  "
              f"{mean_diff:>9.4f}  {stego_path}")

    return results

def test_message_length_scaling(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 8 — SCALABILITÀ CON LUNGHEZZA MESSAGGIO")
    print(f"{'' * 70}")

    y1, x1, y2, x2 = roi_coords
    roi_matrix = image[y1:y2, x1:x2]
    capacity = compute_capacity(roi_matrix, 8, "mid")
    max_chars = min(capacity['max_text_chars'], 500)

    lengths = [1, 5, 10, 20, 50, 100]
    lengths = [l for l in lengths if l <= max_chars]
    if max_chars > 100:
        lengths.append(max_chars)

    results = []

    print(f"Capacità massima: {capacity['max_text_chars']} caratteri")
    print("Block size: 8  SV range: mid  Delta: 15")
    print(f"{'Char':>6}  {'Bit':>6}  {'PSNR':>8}  {'SSIM':>7}  {'BER':>8}  {'OK':>3}  {'Tempo':>6}")
    print(f"{'' * 60}")

    for length in lengths:
        message = "A" * length
        payload_bits = text_to_binary(message)

        t_start = time.time()
        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=15.0,
        )
        extracted_bits = extract_from_full_image(
            stego_image, image, roi_coords,
            block_size=8, sv_range="mid",
            max_bits=len(payload_bits),
        )
        t_elapsed = time.time() - t_start

        extracted_message = binary_to_text(extracted_bits)
        psnr = compute_psnr(image, stego_image)
        ssim = compute_ssim(image, stego_image)
        ber = compute_ber(payload_bits, extracted_bits)
        correct = extracted_message == message

        result = {
            "msg_length": length,
            "payload_bits": len(payload_bits),
            "psnr": psnr,
            "ssim": ssim,
            "ber": ber,
            "correct": correct,
            "time_s": t_elapsed,
        }
        results.append(result)

        ok_str = "" if correct else ""
        print(f"{length:>6}  {len(payload_bits):>6}  {psnr:>7.2f}dB  {ssim:>6.4f}  "
              f"{ber:>8.6f}  {ok_str:>3}  {t_elapsed:>5.1f}s")

    return results

def test_roi_embedding(
    image: np.ndarray,
    output_dir: str = "test_output",
) -> list[dict]:

    print(f"{'' * 70}")
    print("TEST 9 — EMBEDDING CON ROI DI DIVERSE DIMENSIONI")
    print(f"{'' * 70}")

    h, w = image.shape
    message = "Test ROI embedding"
    payload_bits = text_to_binary(message)

    roi_configs = [
        ("100% immagine", (0, 0, h, w)),
        ("75% centrale", (h // 8, w // 8, h * 7 // 8, w * 7 // 8)),
        ("50% centrale", (h // 4, w // 4, h * 3 // 4, w * 3 // 4)),
        ("25% centrale", (h * 3 // 8, w * 3 // 8, h * 5 // 8, w * 5 // 8)),
    ]

    results = []

    print(f"Messaggio: \"{message}\"")
    print(f"Immagine: {h}x{w}")
    print(f"{'ROI':<20}  {'Dim ROI':<10}  {'Capacita':<10}  {'PSNR':<8}  "
          f"{'SSIM':<7}  {'BER':<8}  {'OK':<3}")
    print(f"{'' * 80}")

    for label, roi_coords in roi_configs:
        y1, x1, y2, x2 = roi_coords
        roi_matrix = image[y1:y2, x1:x2]
        capacity = compute_capacity(roi_matrix, 8, "mid")

        if len(payload_bits) > capacity['total_bits']:
            results.append({
                "roi_label": label,
                "roi_size": f"{y2-y1}x{x2-x1}",
                "capacity_bits": capacity['total_bits'],
                "psnr": None,
                "ssim": None,
                "ber": None,
                "correct": False,
                "note": "Capacita insufficiente",
            })
            print(f"{label:<20}  {y2-y1}x{x2-x1:<6}  {capacity['total_bits']:<10}  "
                  f"{'':>8}  {'':>7}  {'':>8}  SKIP")
            continue

        stego_image, embed_info = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=15.0,
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

        outside_modified = not np.array_equal(
            image[:y1, :] if y1 > 0 else image[y2:, :],
            stego_image[:y1, :] if y1 > 0 else stego_image[y2:, :],
        )

        result = {
            "roi_label": label,
            "roi_size": f"{y2-y1}x{x2-x1}",
            "capacity_bits": capacity['total_bits'],
            "psnr": psnr,
            "ssim": ssim,
            "ber": ber,
            "correct": correct,
            "outside_intact": not outside_modified,
            "note": "",
        }
        results.append(result)

        ok_str = "" if correct else ""
        print(f"{label:<20}  {y2-y1}x{x2-x1:<6}  {capacity['total_bits']:<10}  "
              f"{psnr:>7.2f}dB  {ssim:>5.4f}  {ber:>8.6f}  {ok_str}")

    print(f"")
    print("Strategia B (sfondo):")

    bb_center = BoundingBox(w // 4, h // 4, w * 3 // 4, h * 3 // 4, 0.9, 0, "object")
    roi_result = select_roi((h, w), [bb_center], strategy="B")
    bg_roi_coords = (0, 0, h, w)

    bg_capacity = compute_capacity(image, 8, "mid")
    if len(payload_bits) <= bg_capacity['total_bits']:
        stego_bg, _ = embed_in_full_image(
            image, bg_roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=15.0,
        )
        psnr_bg = compute_psnr(image, stego_bg)
        print(f"  Sfondo (escluso box centrale): PSNR={psnr_bg:.2f}dB")

    return results

def test_approximation_metrics(
    image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
) -> list[dict]:
    """
    TEST 10 — METRICHE DI APPROSSIMAZIONE SVD
    Calcola per ogni configurazione (block_size, sv_range, delta):
      1. Errore di approssimazione:
         - Norma di Frobenius: ||A - A_stego||_F
         - Norma Spettrale (2-norma): σ_max(A - A_stego)
      2. Fattore di compressione e occupazione di memoria:
         - Fattore: 2k * (1/m + 1/n)
         - Memoria matrice piena vs. rappresentazione SVD troncata
      3. Numero di condizionamento:
         - κ(A_orig) = σ_1 / σ_n  per la ROI originale
         - κ(A_stego) = σ_1 / σ_n  per la ROI stego
    """

    print(f"{'=' * 70}")
    print("TEST 10 — METRICHE DI APPROSSIMAZIONE SVD")
    print(f"{'=' * 70}")

    message = "Test metriche di approssimazione SVD"
    payload_bits = text_to_binary(message)

    block_sizes = [4, 8, 16]
    sv_ranges = ["first", "mid", "last"]
    deltas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    y1, x1, y2, x2 = roi_coords
    roi_orig = image[y1:y2, x1:x2].copy()
    m_roi, n_roi = roi_orig.shape
    print(f"ROI: {m_roi}×{n_roi} pixel")
    print(f"Messaggio: \"{message}\" ({len(message)} char, {len(payload_bits)} bit)\n")

    U_roi, sigma_roi, Vt_roi = svd_compact(roi_orig)
    k_roi = len(sigma_roi)
    kappa_orig_global = float(sigma_roi[0] / sigma_roi[-1]) if sigma_roi[-1] > 1e-15 else float('inf')
    mem_full = m_roi * n_roi
    mem_svd = k_roi * (m_roi + n_roi + 1)
    compression_ratio_global = mem_full / mem_svd if mem_svd > 0 else float('inf')
    compression_factor_global = 2.0 * k_roi * (1.0 / m_roi + 1.0 / n_roi)

    print("— Metriche globali ROI (SVD compatta, senza embedding) —")
    print(f"  Rango effettivo k      : {k_roi}")
    print(f"  σ_1                    : {sigma_roi[0]:.4f}")
    print(f"  σ_k                    : {sigma_roi[-1]:.4f}")
    print(f"  κ(ROI_orig)            : {kappa_orig_global:.4f}")
    print(f"  Memoria matrice piena  : {mem_full:,} scalari")
    print(f"  Memoria SVD troncata   : {mem_svd:,} scalari  (k={k_roi})")
    print(f"  Rapporto compressione  : {compression_ratio_global:.2f}×")
    print(f"  Fattore 2k(1/m+1/n)   : {compression_factor_global:.6f}")
    print()

    results = []

    print(f"{'#':>4}  {'Block':>5}  {'SV Range':>8}  {'Delta':>5}  "
          f"{'||·||_F':>10}  {'||·||_2':>10}  "
          f"{'κ_orig':>10}  {'κ_stego':>10}  "
          f"{'CF':>8}  {'Mem_full':>9}  {'Mem_svd':>9}")
    print(f"{'-' * 110}")

    count = 0
    for block_size in block_sizes:
        for sv_range in sv_ranges:
            for delta in deltas:
                count += 1

                capacity = compute_capacity(roi_orig, block_size, sv_range)
                if len(payload_bits) > capacity['total_bits']:
                    results.append({
                        "block_size": block_size,
                        "sv_range": sv_range,
                        "delta": delta,
                        "frobenius_norm": None,
                        "spectral_norm": None,
                        "kappa_orig": None,
                        "kappa_stego": None,
                        "compression_factor": None,
                        "mem_full_scalars": None,
                        "mem_svd_scalars": None,
                        "note": "Capacità insufficiente",
                    })
                    print(f"{count:>4}  {block_size:>5}  {sv_range:>8}  {delta:>5.0f}  "
                          f"{'':>10}  {'':>10}  "
                          f"{'':>10}  {'':>10}  "
                          f"{'':>8}  {'':>9}  {'SKIP':>9}")
                    continue

                stego_image, _ = embed_in_full_image(
                    image, roi_coords, payload_bits,
                    block_size=block_size, sv_range=sv_range, delta=delta,
                )
                roi_stego = stego_image[y1:y2, x1:x2].copy()

                diff_matrix = roi_orig - roi_stego
                frobenius_norm = float(np.sqrt(np.sum(diff_matrix ** 2)))

                _, sigma_diff, _ = svd_compact(diff_matrix)
                spectral_norm = float(sigma_diff[0]) if len(sigma_diff) > 0 else 0.0

                k_block = block_size
                mem_block_full = block_size * block_size
                mem_block_svd = k_block * (block_size + block_size + 1)
                compression_factor = 2.0 * k_block * (1.0 / block_size + 1.0 / block_size)

                n_blocks = capacity['n_blocks']
                mem_total_full = n_blocks * mem_block_full
                mem_total_svd = n_blocks * mem_block_svd

                from src.image_utils import split_into_blocks as _split
                orig_blocks, _, _ = _split(roi_orig, block_size)
                stego_blocks, _, _ = _split(roi_stego, block_size)

                kappa_orig_list = []
                kappa_stego_list = []
                for ob, sb in zip(orig_blocks, stego_blocks):
                    _, s_o, _ = svd_compact(ob)
                    _, s_s, _ = svd_compact(sb)
                    if len(s_o) > 0 and s_o[-1] > 1e-15:
                        kappa_orig_list.append(s_o[0] / s_o[-1])
                    else:
                        kappa_orig_list.append(float('inf'))
                    if len(s_s) > 0 and s_s[-1] > 1e-15:
                        kappa_stego_list.append(s_s[0] / s_s[-1])
                    else:
                        kappa_stego_list.append(float('inf'))

                finite_ko = [k for k in kappa_orig_list if k != float('inf')]
                finite_ks = [k for k in kappa_stego_list if k != float('inf')]
                kappa_orig_mean = float(np.mean(finite_ko)) if finite_ko else float('inf')
                kappa_stego_mean = float(np.mean(finite_ks)) if finite_ks else float('inf')

                result = {
                    "block_size": block_size,
                    "sv_range": sv_range,
                    "delta": delta,
                    "frobenius_norm": frobenius_norm,
                    "spectral_norm": spectral_norm,
                    "kappa_orig": kappa_orig_mean,
                    "kappa_stego": kappa_stego_mean,
                    "compression_factor": compression_factor,
                    "mem_full_scalars": mem_total_full,
                    "mem_svd_scalars": mem_total_svd,
                    "note": "",
                }
                results.append(result)

                ko_str = f"{kappa_orig_mean:.2f}" if kappa_orig_mean != float('inf') else "∞"
                ks_str = f"{kappa_stego_mean:.2f}" if kappa_stego_mean != float('inf') else "∞"
                print(f"{count:>4}  {block_size:>5}  {sv_range:>8}  {delta:>5.0f}  "
                      f"{frobenius_norm:>10.4f}  {spectral_norm:>10.4f}  "
                      f"{ko_str:>10}  {ks_str:>10}  "
                      f"{compression_factor:>8.4f}  {mem_total_full:>9,}  {mem_total_svd:>9,}")


    valid = [r for r in results if r["frobenius_norm"] is not None]
    skipped = len(results) - len(valid)
    print(f"\n{'=' * 70}")
    print(f"Riepilogo Test 10: {len(valid)} configurazioni valide, {skipped} saltate")
    if valid:
        avg_frob = np.mean([r["frobenius_norm"] for r in valid])
        avg_spec = np.mean([r["spectral_norm"] for r in valid])
        print(f"  ||A-A_stego||_F  medio : {avg_frob:.4f}")
        print(f"  ||A-A_stego||_2  medio : {avg_spec:.4f}")
        finite_kappas = [r["kappa_orig"] for r in valid if r["kappa_orig"] != float('inf')]
        if finite_kappas:
            print(f"  κ(A_orig) medio        : {np.mean(finite_kappas):.2f}")
    print(f"{'=' * 70}\n")

    return results


def save_results_csv(results: list[dict], filename: str, output_dir: str = "test_output") -> str:

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

def main():
    parser = argparse.ArgumentParser(description="Suite di test per la steganografia SVD")
    parser.add_argument("-i", "--image", type=str, default=None,
                        help="Percorso immagine di test (se omesso, usa immagine sintetica)")
    parser.add_argument("-o", "--output", type=str, default="test_output",
                        help="Directory di output per risultati e CSV (default: test_output)")
    parser.add_argument("-t", "--test", type=int, default=None,
                help="Esegui solo il test specificato (1-10). Se omesso, esegue tutti.")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Errore: file '{args.image}' non trovato!")
            sys.exit(1)
        print(f"Caricamento immagine: {args.image}")
        image = load_image_as_matrix(args.image, grayscale=True)
    else:
        print("Generazione immagine sintetica di test (64x64)...")
        image = generate_test_image(64, 64)
        save_image(image, os.path.join(output_dir, "test_image_synthetic.png"))

    h, w = image.shape
    print(f"Dimensioni: {h}×{w}")

    roi_coords = (0, 0, h, w)
    print(f"ROI: intera immagine ({h}×{w})")

    print(f"{'' * 70}")
    print("SUITE DI TEST — STEGANOGRAFIA SVD")
    print(f"Immagine: {'sintetica' if not args.image else args.image}")
    print(f"Output: {output_dir}/")
    print(f"{'' * 70}")

    all_tests = {
        1: ("Sweep parametri", test_parameter_sweep, "results_sweep.csv"),
        2: ("Impatto delta", test_delta_impact, "results_delta.csv"),
        3: ("Analisi capacità", test_capacity_analysis, "results_capacity.csv"),
        4: ("Robustezza rumore", test_robustness_noise, "results_noise.csv"),
        5: ("Robustezza blur", test_robustness_blur, "results_blur.csv"),
        6: ("Robustezza JPEG", test_robustness_jpeg, "results_jpeg.csv"),
        7: ("Impatto visivo SV", test_sv_range_visual_impact, "results_sv_visual.csv"),
        8: ("Scalabilita messaggio", test_message_length_scaling, "results_scaling.csv"),
        9: ("Embedding con ROI", test_roi_embedding, "results_roi.csv"),
        10: ("Metriche approssimazione", test_approximation_metrics, "results_approx_metrics.csv"),
    }

    tests_to_run = [args.test] if args.test else list(all_tests.keys())

    for test_id in tests_to_run:
        if test_id not in all_tests:
            print(f"Test {test_id} non esiste. Disponibili: 1-10")
            continue

        name, func, csv_file = all_tests[test_id]

        if test_id in (1, 7):
            results = func(image, roi_coords, output_dir)
        elif test_id == 9:
            results = func(image, output_dir)
        else:
            results = func(image, roi_coords)

        if results:
            csv_path = save_results_csv(results, csv_file, output_dir)
            print(f"Risultati salvati: {csv_path}")

    print(f"{'' * 70}")
    print("SUITE DI TEST COMPLETATA")
    print(f"{'' * 70}")
    print(f"Output salvati in: {output_dir}/")
    print(f"Test eseguiti: {len(tests_to_run)}")
    print(f"{'' * 70}\n")

if __name__ == "__main__":
    main()