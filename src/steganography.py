import numpy as np
from src.svd import svd_compact, reconstruct
from src.image_utils import split_into_blocks, merge_blocks

def text_to_binary(message: str) -> np.ndarray:

    bits = []
    for ch in message:
        code = ord(ch)
        if code > 127:
            raise ValueError(
                f"Carattere non-ASCII trovato: '{ch}' (codice {code}). "
                f"Usa solo caratteri ASCII (0-127)."
            )
        for i in range(6, -1, -1):
            bits.append((code >> i) & 1)

    bits.extend([0] * 7)
    return np.array(bits, dtype=np.int32)

def binary_to_text(bits: np.ndarray) -> str:

    if len(bits) % 7 != 0:
        bits = bits[:len(bits) - (len(bits) % 7)]

    chars = []
    for i in range(0, len(bits), 7):
        char_bits = bits[i:i + 7]
        char_val = 0
        for bit in char_bits:
            char_val = (char_val << 1) | int(bit)

        if char_val == 0:
            break
        chars.append(chr(char_val))

    return ''.join(chars)

def image_to_binary(image_matrix: np.ndarray) -> np.ndarray:

    h, w = image_matrix.shape[:2]
    pixels = np.clip(image_matrix, 0, 255).astype(np.uint8).flatten()

    bits = []
    for dim in [h, w]:
        for i in range(15, -1, -1):
            bits.append((dim >> i) & 1)
    for px in pixels:
        for i in range(7, -1, -1):
            bits.append((px >> i) & 1)

    return np.array(bits, dtype=np.int32)

def binary_to_image(bits: np.ndarray) -> np.ndarray:

    h = 0
    for i in range(16):
        h = (h << 1) | int(bits[i])
    w = 0
    for i in range(16, 32):
        w = (w << 1) | int(bits[i])

    pixel_bits = bits[32:]
    n_pixels = h * w
    pixels = []
    for i in range(0, min(n_pixels * 8, len(pixel_bits)), 8):
        byte_val = 0
        for bit in pixel_bits[i:i + 8]:
            byte_val = (byte_val << 1) | int(bit)
        pixels.append(byte_val)

    pixels = np.array(pixels[:n_pixels], dtype=np.uint8)
    if len(pixels) < n_pixels:
        pixels = np.pad(pixels, (0, n_pixels - len(pixels)))

    return pixels.reshape(h, w).astype(np.float64)

def compute_capacity(
    roi_matrix: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
) -> dict:

    h, w = roi_matrix.shape[:2]

    h_blocks = int(np.ceil(h / block_size))
    w_blocks = int(np.ceil(w / block_size))
    n_blocks = h_blocks * w_blocks

    sv_per_block = block_size
    bits_per_block = _get_sv_indices_count(sv_per_block, sv_range)

    total_bits = n_blocks * bits_per_block
    max_text_chars = (total_bits // 7) - 1

    return {
        "roi_shape": (h, w),
        "block_size": block_size,
        "n_blocks": n_blocks,
        "sv_per_block": sv_per_block,
        "bits_per_block": bits_per_block,
        "total_bits": total_bits,
        "max_text_chars": max_text_chars,
        "sv_range": sv_range,
    }

def _get_sv_indices_count(n_sv: int, sv_range: str) -> int:

    return len(_get_sv_indices(n_sv, sv_range))

def _get_sv_indices(n_sv: int, sv_range: str) -> list[int]:

    sv_range = sv_range.lower()

    if n_sv <= 2:
        return list(range(n_sv))

    if sv_range == "first":
        end = max(1, n_sv // 3)
        return list(range(0, end))
    elif sv_range == "last":
        start = n_sv - max(1, n_sv // 3)
        return list(range(start, n_sv))
    elif sv_range == "mid":
        start = max(1, n_sv // 3)
        end = n_sv - max(1, n_sv // 3)
        if start >= end:
            mid = n_sv // 2
            return [mid]
        return list(range(start, end))
    else:
        raise ValueError(f"sv_range non valido: '{sv_range}'. Usa 'first', 'mid', o 'last'.")

def _embed_bit(sigma_val: float, bit: int, delta: float) -> float:

    if bit == 1:
        return sigma_val + delta
    else:
        return max(0.0, sigma_val - delta)

def _extract_bit(sigma_stego: float, sigma_original: float) -> int:

    return 1 if sigma_stego > sigma_original else 0

def embed_message(
    roi_matrix: np.ndarray,
    payload_bits: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
) -> tuple[np.ndarray, dict]:

    blocks, positions, original_shape = split_into_blocks(roi_matrix, block_size)

    total_payload_bits = len(payload_bits)
    bit_index = 0
    bits_embedded = 0

    modified_blocks = []

    for block in blocks:
        U_b, sigma_b, Vt_b = svd_compact(block)
        n_sv = len(sigma_b)
        sv_indices = _get_sv_indices(n_sv, sv_range)

        sigma_modified = sigma_b.copy()
        for sv_idx in sv_indices:
            if bit_index >= total_payload_bits:
                break

            bit = int(payload_bits[bit_index])
            sigma_modified[sv_idx] = _embed_bit(sigma_b[sv_idx], bit, delta)
            bit_index += 1
            bits_embedded += 1

        block_recon = reconstruct(U_b, sigma_modified, Vt_b)
        modified_blocks.append(block_recon)

    stego_roi = merge_blocks(modified_blocks, positions, original_shape, block_size)

    capacity = compute_capacity(roi_matrix, block_size, sv_range)
    embed_info = {
        "bits_embedded": bits_embedded,
        "total_payload_bits": total_payload_bits,
        "capacity_bits": capacity["total_bits"],
        "utilization_pct": bits_embedded / capacity["total_bits"] * 100 if capacity["total_bits"] > 0 else 0,
        "block_size": block_size,
        "sv_range": sv_range,
        "delta": delta,
        "n_blocks": len(blocks),
        "payload_fully_embedded": bit_index >= total_payload_bits,
    }

    if not embed_info["payload_fully_embedded"]:
        print(f"ATTENZIONE: il payload ({total_payload_bits} bit) supera la capacità "
              f"({capacity['total_bits']} bit). Sono stati incorporati solo {bits_embedded} bit.")

    return stego_roi, embed_info

def extract_message(
    stego_roi_matrix: np.ndarray,
    original_roi_matrix: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    max_bits: int | None = None,
) -> np.ndarray:

    stego_blocks, positions, original_shape = split_into_blocks(stego_roi_matrix, block_size)
    orig_blocks, _, _ = split_into_blocks(original_roi_matrix, block_size)

    extracted_bits = []

    for stego_block, orig_block in zip(stego_blocks, orig_blocks):
        U_o, sigma_o, Vt_o = svd_compact(orig_block)
        n_sv = len(sigma_o)
        sv_indices = _get_sv_indices(n_sv, sv_range)

        V_o = Vt_o.T
        projected = U_o.T @ stego_block @ V_o
        sigma_projected = np.diag(projected)

        for sv_idx in sv_indices:
            if sv_idx < len(sigma_projected):
                bit = _extract_bit(sigma_projected[sv_idx], sigma_o[sv_idx])
            else:
                bit = 0
            extracted_bits.append(bit)

            if max_bits is not None and len(extracted_bits) >= max_bits:
                return np.array(extracted_bits, dtype=np.int32)

    return np.array(extracted_bits, dtype=np.int32)

def embed_in_full_image(
    image_matrix: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    payload_bits: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
) -> tuple[np.ndarray, dict]:

    y1, x1, y2, x2 = roi_coords
    roi = image_matrix[y1:y2, x1:x2].copy()

    stego_roi, embed_info = embed_message(roi, payload_bits, block_size, sv_range, delta)

    stego_image = image_matrix.copy()
    stego_image[y1:y2, x1:x2] = stego_roi
    embed_info["roi_coords"] = roi_coords

    return stego_image, embed_info

def extract_from_full_image(
    stego_image: np.ndarray,
    original_image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    block_size: int = 8,
    sv_range: str = "mid",
    max_bits: int | None = None,
) -> np.ndarray:

    y1, x1, y2, x2 = roi_coords
    stego_roi = stego_image[y1:y2, x1:x2].copy()
    original_roi = original_image[y1:y2, x1:x2].copy()

    return extract_message(stego_roi, original_roi, block_size, sv_range, max_bits)

def print_embed_report(embed_info: dict) -> None:

    sv_names = {
        "first": "Primi (massima robustezza, artefatti visibili)",
        "mid": "Intermedi (miglior compromesso)",
        "last": "Ultimi (invisibile, vulnerabile a JPEG)",
    }

    print(f"{'' * 60}")
    print("REPORT EMBEDDING")
    print(f"{'' * 60}")
    print("Parametri:")
    print(f"Block size:          {embed_info['block_size']}×{embed_info['block_size']}")
    print(f"Valori singolari:    {embed_info['sv_range']} — {sv_names.get(embed_info['sv_range'], '?')}")
    print(f"Delta:               {embed_info['delta']}")
    print(f"Numero blocchi:      {embed_info['n_blocks']}")
    print("Payload:")
    print(f"Bit del payload:     {embed_info['total_payload_bits']}")
    print(f"Bit incorporati:     {embed_info['bits_embedded']}")
    print(f"Capacità totale:     {embed_info['capacity_bits']} bit")
    print(f"Utilizzo capacità:   {embed_info['utilization_pct']:.1f}%")

    if embed_info["payload_fully_embedded"]:
        print("Payload incorporato con successo!")
    else:
        print("Payload TRONCATO! Capacità insufficiente.")

    if "roi_coords" in embed_info:
        y1, x1, y2, x2 = embed_info["roi_coords"]
        print("ROI: ({x1},{y1})→({x2},{y2})")

    print(f"{'' * 60}")