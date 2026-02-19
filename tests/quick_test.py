import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.svd import svd_compact, reconstruct
from src.image_utils import split_into_blocks, merge_blocks
from src.steganography import (
    text_to_binary,
    binary_to_text,
    embed_in_full_image,
    extract_from_full_image,
    compute_capacity,
)
from src.image_utils import compute_psnr
from src.yolo_roi import BoundingBox, select_roi, extract_roi_region

def print_header(title):
    print(f"{'' * 60}")
    print(f"{title}")
    print(f"{'' * 60}")

def print_result(name, passed, detail=""):
    icon = "" if passed else ""
    detail_str = f" — {detail}" if detail else ""
    print(f"{icon} {name}{detail_str}")
    return passed

def test_svd_correctness():
    print_header("TEST 1 — Correttezza SVD custom vs numpy")
    all_passed = True

    test_matrices = {
        "Matrice 4×4 random": np.random.default_rng(42).standard_normal((4, 4)) * 50 + 128,
        "Matrice 8×8 random": np.random.default_rng(43).standard_normal((8, 8)) * 50 + 128,
        "Matrice 8×8 gradiente": np.array([[i + j for j in range(8)] for i in range(8)], dtype=np.float64) * 30,
        "Matrice 4×8 rettangolare": np.random.default_rng(44).standard_normal((4, 8)) * 50 + 128,
        "Matrice 8×4 rettangolare": np.random.default_rng(45).standard_normal((8, 4)) * 50 + 128,
        "Matrice identità 4×4": np.eye(4) * 100.0,
    }

    for label, X in test_matrices.items():

        U, sigma, Vt = svd_compact(X)

        U_np, sigma_np, Vt_np = np.linalg.svd(X, full_matrices=False)

        k = min(len(sigma), len(sigma_np))
        sv_error = np.max(np.abs(sigma[:k] - sigma_np[:k]))

        X_recon = reconstruct(U, sigma, Vt)
        recon_error = np.linalg.norm(X - X_recon, 'fro') / np.linalg.norm(X, 'fro')

        UtU = U.T @ U
        orth_U_error = np.max(np.abs(UtU - np.eye(U.shape[1])))

        VVt = Vt @ Vt.T
        orth_V_error = np.max(np.abs(VVt - np.eye(Vt.shape[0])))

        passed = sv_error < 0.1 and recon_error < 1e-5 and orth_U_error < 1e-8 and orth_V_error < 1e-8
        all_passed &= print_result(
            label,
            passed,
            f"SV err={sv_error:.2e}, recon err={recon_error:.2e}, "
            f"orth U={orth_U_error:.2e}, orth V={orth_V_error:.2e}"
        )

    return all_passed

def test_text_binary_conversion():
    print_header("TEST 2 — Conversione testo ↔ binario")
    all_passed = True

    test_messages = [
        "Hello!",
        "A",
        "Test 123",
        "SVD steganography",
        "Messaggio con spazi e punteggiatura!",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
        "",
    ]

    for msg in test_messages:
        bits = text_to_binary(msg)
        recovered = binary_to_text(bits)
        passed = recovered == msg
        all_passed &= print_result(
            f"\"{msg}\"",
            passed,
            f"{'OK' if passed else f'recuperato: \"{recovered}\"'} ({len(bits)} bit)"
        )

    return all_passed

def test_embed_extract_roundtrip():
    print_header("TEST 3 — Embedding + Extraction roundtrip")
    all_passed = True

    rng = np.random.default_rng(42)
    image = rng.standard_normal((32, 32)) * 40 + 128
    image = np.clip(image, 0, 255)

    roi_coords = (0, 0, 32, 32)

    test_cases = [

        (8, "mid", 15.0, "Hi"),
        (8, "first", 15.0, "AB"),
        (8, "last", 15.0, "XY"),
        (8, "mid", 10.0, "Test"),
        (8, "mid", 25.0, "Ok"),
        (4, "mid", 15.0, "Hello!"),
        (16, "mid", 15.0, "A"),
    ]

    for block_size, sv_range, delta, message in test_cases:
        payload_bits = text_to_binary(message)

        roi_matrix = image[0:32, 0:32]
        capacity = compute_capacity(roi_matrix, block_size, sv_range)

        if len(payload_bits) > capacity['total_bits']:
            print_result(
                f"bs={block_size}, sv={sv_range}, δ={delta}, \"{message}\"",
                True,
                f"SKIP (capacità {capacity['total_bits']} bit < {len(payload_bits)} bit)"
            )
            continue

        t0 = time.time()

        stego_image, embed_info = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=block_size, sv_range=sv_range, delta=delta,
        )

        extracted_bits = extract_from_full_image(
            stego_image, image, roi_coords,
            block_size=block_size, sv_range=sv_range,
            max_bits=len(payload_bits),
        )

        extracted_message = binary_to_text(extracted_bits)
        elapsed = time.time() - t0

        psnr = compute_psnr(image, stego_image)
        ber = np.sum(payload_bits[:len(extracted_bits)] != extracted_bits[:len(payload_bits)]) / len(payload_bits)
        correct = extracted_message == message

        all_passed &= print_result(
            f"bs={block_size}, sv={sv_range}, δ={delta}, \"{message}\"",
            correct,
            f"{'OK' if correct else f'estratto: \"{extracted_message}\"'}  "
            f"PSNR={psnr:.1f}dB  BER={ber:.4f}  {elapsed:.1f}s"
        )

    return all_passed

def test_visual_quality():
    print_header("TEST 4 — Qualità visiva (PSNR)")
    all_passed = True

    rng = np.random.default_rng(42)
    image = rng.standard_normal((32, 32)) * 40 + 128
    image = np.clip(image, 0, 255)

    roi_coords = (0, 0, 32, 32)
    message = "Hi"
    payload_bits = text_to_binary(message)

    for delta in [5.0, 10.0, 15.0, 20.0, 30.0]:
        stego_image, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range="mid", delta=delta,
        )
        psnr = compute_psnr(image, stego_image)

        passed = psnr > 15.0
        all_passed &= print_result(
            f"δ={delta:5.1f}",
            passed,
            f"PSNR = {psnr:.2f} dB {'(buono)' if psnr > 40 else '(accettabile)' if psnr > 30 else '(basso)'}"
        )

    return all_passed

def test_edge_cases():
    print_header("TEST 5 — Edge cases")
    all_passed = True

    image_uniform = np.ones((32, 32), dtype=np.float64) * 128
    roi_coords_unif = (0, 0, 32, 32)
    payload = text_to_binary("A")

    try:
        stego, info = embed_in_full_image(
            image_uniform, roi_coords_unif, payload,
            block_size=8, sv_range="first", delta=15.0,
        )
        extracted_bits = extract_from_full_image(
            stego, image_uniform, roi_coords_unif,
            block_size=8, sv_range="first",
            max_bits=len(payload),
        )
        extracted = binary_to_text(extracted_bits)
        all_passed &= print_result(
            "Immagine uniforme (first, 32×32)",
            extracted == "A",
            f"estratto: \"{extracted}\" — blocchi rango-1, 1 bit/blocco"
        )
    except Exception as e:
        all_passed &= print_result("Immagine uniforme", False, f"ERRORE: {e}")

    rng = np.random.default_rng(42)
    image_extreme = rng.choice([0.0, 255.0], size=(16, 16))
    roi_coords = (0, 0, 16, 16)
    try:
        stego, _ = embed_in_full_image(
            image_extreme, roi_coords, payload,
            block_size=8, sv_range="mid", delta=15.0,
        )
        extracted_bits = extract_from_full_image(
            stego, image_extreme, roi_coords,
            block_size=8, sv_range="mid",
            max_bits=len(payload),
        )
        extracted = binary_to_text(extracted_bits)
        all_passed &= print_result("Immagine valori estremi (0/255)", extracted == "A", f"estratto: \"{extracted}\"")
    except Exception as e:
        all_passed &= print_result("Immagine valori estremi", False, f"ERRORE: {e}")

    rng2 = np.random.default_rng(42)
    image_cap = rng2.standard_normal((16, 16)) * 40 + 128
    image_cap = np.clip(image_cap, 0, 255)
    cap = compute_capacity(image_cap, block_size=8, sv_range="mid")
    all_passed &= print_result(
        "Calcolo capacità 16×16 bs=8 mid",
        cap['total_bits'] > 0 and cap['n_blocks'] == 4,
        f"{cap['n_blocks']} blocchi, {cap['bits_per_block']} bit/blocco, "
        f"{cap['total_bits']} bit totali, {cap['max_text_chars']} chars max"
    )

    rng3 = np.random.default_rng(42)
    test_matrix = rng3.standard_normal((17, 17)) * 50 + 128
    blocks, positions, orig_shape = split_into_blocks(test_matrix, 8)
    merged = merge_blocks(blocks, positions, orig_shape, 8)
    merge_ok = np.allclose(test_matrix, merged)
    all_passed &= print_result(
        "Split/merge con dimensioni nonmultiple (17×17, bs=8)",
        merge_ok,
        f"{'OK' if merge_ok else 'ERRORE: matrici diverse'}"
    )

    return all_passed

def test_sv_ranges():
    print_header("TEST 6 — Consistenza dei SV range")
    all_passed = True

    rng = np.random.default_rng(42)
    image = rng.standard_normal((32, 32)) * 40 + 128
    image = np.clip(image, 0, 255)
    roi_coords = (0, 0, 32, 32)

    for sv_range in ["first", "mid", "last"]:
        message = "Hi"
        payload_bits = text_to_binary(message)

        capacity = compute_capacity(image, 8, sv_range)

        if len(payload_bits) > capacity['total_bits']:
            print_result(f"sv_range='{sv_range}'", True, "SKIP (capacità insufficiente)")
            continue

        stego, _ = embed_in_full_image(
            image, roi_coords, payload_bits,
            block_size=8, sv_range=sv_range, delta=15.0,
        )
        extracted_bits = extract_from_full_image(
            stego, image, roi_coords,
            block_size=8, sv_range=sv_range,
            max_bits=len(payload_bits),
        )
        extracted = binary_to_text(extracted_bits)
        psnr = compute_psnr(image, stego)

        all_passed &= print_result(
            f"sv_range='{sv_range}'",
            extracted == message,
            f"{'OK' if extracted == message else f'estratto: \"{extracted}\"'}  "
            f"cap={capacity['total_bits']}bit  PSNR={psnr:.1f}dB"
        )

    return all_passed

def test_roi_selection():
    print_header("TEST 7 — Selezione ROI")
    all_passed = True

    bb1 = BoundingBox(10, 20, 50, 80, 0.95, 0, "person")
    all_passed &= print_result(
        "BoundingBox properties",
        bb1.width == 40 and bb1.height == 60 and bb1.area == 2400,
        f"width={bb1.width}, height={bb1.height}, area={bb1.area}"
    )

    bb_large = BoundingBox(10, 10, 90, 90, 0.9, 0, "person")
    bb_small = BoundingBox(50, 50, 70, 70, 0.8, 1, "cat")
    bboxes = [bb_large, bb_small]
    image_shape = (100, 100)

    roi_a = select_roi(image_shape, bboxes, strategy="A")
    all_passed &= print_result(
        "Strategia A (soggetto, box piu grande)",
        roi_a.selected_box == bb_large and roi_a.strategy == "A",
        f"box={roi_a.selected_box.class_name}, area={roi_a.selected_box.area}"
    )

    roi_a2 = select_roi(image_shape, bboxes, strategy="A", box_index=1)
    all_passed &= print_result(
        "Strategia A (box_index=1)",
        roi_a2.selected_box == bb_small,
        f"box={roi_a2.selected_box.class_name}, area={roi_a2.selected_box.area}"
    )

    roi_b = select_roi(image_shape, bboxes, strategy="B")
    inside_large = roi_b.mask[bb_large.y1:bb_large.y2, bb_large.x1:bb_large.x2]
    all_passed &= print_result(
        "Strategia B (sfondo, esclude i box)",
        not np.any(inside_large) and roi_b.strategy == "B",
        f"pixel ROI={roi_b.roi_pixel_count}"
    )

    roi_c = select_roi(image_shape, bboxes, strategy="C")
    all_passed &= print_result(
        "Strategia C (automatico, box piu grande)",
        roi_c.selected_box == bb_large and roi_c.strategy == "C",
        f"box={roi_c.selected_box.class_name}, area={roi_c.selected_box.area}"
    )

    roi_empty = select_roi(image_shape, [], strategy="A")
    all_passed &= print_result(
        "Nessuna detection (fallback intera immagine)",
        np.all(roi_empty.mask) and roi_empty.selected_box is None,
        f"pixel ROI={roi_empty.roi_pixel_count} (atteso {100 * 100})"
    )

    rng = np.random.default_rng(42)
    image_matrix = rng.standard_normal((100, 100)) * 40 + 128
    roi_region = extract_roi_region(image_matrix, roi_c)
    expected_shape = (bb_large.y2 - bb_large.y1, bb_large.x2 - bb_large.x1)
    all_passed &= print_result(
        "extract_roi_region (con selected_box)",
        roi_region.shape == expected_shape,
        f"shape={roi_region.shape}, atteso={expected_shape}"
    )

    rng2 = np.random.default_rng(42)
    image = rng2.standard_normal((64, 64)) * 40 + 128
    image = np.clip(image, 0, 255)

    bb_roi = BoundingBox(8, 8, 56, 56, 0.95, 0, "object")
    roi_result = select_roi((64, 64), [bb_roi], strategy="C")
    bb = roi_result.selected_box
    roi_coords = (bb.y1, bb.x1, bb.y2, bb.x2)

    message = "ROI test"
    payload_bits = text_to_binary(message)

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

    all_passed &= print_result(
        "Integrazione ROI + embed/extract",
        extracted_message == message,
        f"{'OK' if extracted_message == message else f'estratto: "{extracted_message}"'}  "
        f"PSNR={psnr:.1f}dB"
    )

    outside_unchanged = (
        np.array_equal(image[:8, :], stego_image[:8, :]) and
        np.array_equal(image[56:, :], stego_image[56:, :])
    )
    all_passed &= print_result(
        "Pixel fuori dalla ROI invariati",
        outside_unchanged,
        "OK" if outside_unchanged else "pixel modificati fuori ROI"
    )

    return all_passed

if __name__ == "__main__":
    print(f"{'' * 60}")
    print("TEST RAPIDO DI CORRETTEZZA — Steganografia SVD")
    print(f"{'' * 60}")

    t_start = time.time()
    results = []

    results.append(("SVD correttezza", test_svd_correctness()))
    results.append(("Testo ↔ binario", test_text_binary_conversion()))
    results.append(("Embed/Extract roundtrip", test_embed_extract_roundtrip()))
    results.append(("Qualità visiva", test_visual_quality()))
    results.append(("Edge cases", test_edge_cases()))
    results.append(("SV ranges", test_sv_ranges()))
    results.append(("Selezione ROI", test_roi_selection()))

    elapsed = time.time() - t_start

    print_header("RIEPILOGO")
    total = len(results)
    passed = sum(1 for _, p in results if p)
    for name, p in results:
        icon = "" if p else ""
        print(f"{icon} {name}")

    print(f"{'' * 40}")
    print(f"Test superati: {passed}/{total}")
    print(f"Tempo totale:  {elapsed:.1f}s")

    if passed == total:
        print("TUTTI I TEST SUPERATI!")
    else:
        print(f"{total - passed} test falliti.")

    print(f"{'' * 60}\n")