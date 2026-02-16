"""
Fase 3 — Embedding del Messaggio tramite SVD.

Modulo per nascondere un payload (testo o dati binari) all'interno dei
valori singolari dell'immagine, blocco per blocco.

Algoritmo di embedding — Quantization Index Modulation (QIM):
    Per ogni bit del messaggio, si seleziona un valore singolare σ_i e si
    modifica in modo che sia "pari" (bit=0) o "dispari" (bit=1) rispetto
    a un passo di quantizzazione Δ (delta). Questo approccio:
    - Non richiede l'immagine originale per l'estrazione (blind extraction)
    - È robusto a piccole perturbazioni
    - È matematicamente deducibile

Strategia di selezione dei valori singolari:
    - 'first'  : modifica i primi SV → massima robustezza, ma artefatti visibili
    - 'mid'    : modifica i SV intermedi → miglior compromesso (consigliato)
    - 'last'   : modifica gli ultimi SV → invisibile, ma fragile a JPEG
"""

import numpy as np
from src.svd import svd_compact, reconstruct
from src.image_utils import split_into_blocks, merge_blocks


# ─────────────────────────────────────────────────────────────
#  Conversione Messaggio ↔ Binario
# ─────────────────────────────────────────────────────────────

def text_to_binary(message: str) -> np.ndarray:
    """
    Converte un messaggio di testo in una sequenza binaria (array di 0 e 1).

    Ogni carattere viene codificato in 8 bit (ASCII/UTF-8).
    Viene aggiunto un terminatore speciale (8 bit a zero: 0x00) per
    delimitare la fine del messaggio in fase di estrazione.

    Parametri
    ---------
    message : str
        Il messaggio segreto da nascondere.

    Ritorna
    -------
    bits : np.ndarray
        Array di int (0 o 1) che rappresenta il messaggio + terminatore.
    """
    # Codifica in bytes UTF-8
    msg_bytes = message.encode('utf-8')

    # Aggiungi il byte terminatore 0x00
    msg_bytes += b'\x00'

    bits = []
    for byte in msg_bytes:
        for i in range(7, -1, -1):  # MSB first
            bits.append((byte >> i) & 1)

    return np.array(bits, dtype=np.int32)


def binary_to_text(bits: np.ndarray) -> str:
    """
    Converte una sequenza binaria in testo, fermandosi al byte terminatore (0x00).

    Parametri
    ---------
    bits : np.ndarray
        Array di int (0 o 1).

    Ritorna
    -------
    message : str
        Il messaggio decodificato.
    """
    if len(bits) % 8 != 0:
        # Tronca ai multipli di 8
        bits = bits[:len(bits) - (len(bits) % 8)]

    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte_val = 0
        for bit in byte_bits:
            byte_val = (byte_val << 1) | int(bit)

        if byte_val == 0:  # Terminatore trovato
            break
        chars.append(chr(byte_val))

    return ''.join(chars)


def image_to_binary(image_matrix: np.ndarray) -> np.ndarray:
    """
    Converte un'immagine (matrice numpy) in una sequenza binaria.

    Ogni pixel (uint8, 0-255) viene codificato in 8 bit.
    Viene aggiunto un header con le dimensioni dell'immagine (altezza e larghezza,
    ciascuno codificato in 16 bit) per la ricostruzione.

    Parametri
    ---------
    image_matrix : np.ndarray
        Matrice 2D dell'immagine (grayscale, valori 0-255).

    Ritorna
    -------
    bits : np.ndarray
        Array binario: [16 bit altezza] + [16 bit larghezza] + [pixel in 8 bit ciascuno].
    """
    h, w = image_matrix.shape[:2]
    pixels = np.clip(image_matrix, 0, 255).astype(np.uint8).flatten()

    bits = []

    # Header: altezza (16 bit) + larghezza (16 bit)
    for dim in [h, w]:
        for i in range(15, -1, -1):
            bits.append((dim >> i) & 1)

    # Pixel data
    for px in pixels:
        for i in range(7, -1, -1):
            bits.append((px >> i) & 1)

    return np.array(bits, dtype=np.int32)


def binary_to_image(bits: np.ndarray) -> np.ndarray:
    """
    Ricostruisce un'immagine da una sequenza binaria (con header dimensioni).

    Parametri
    ---------
    bits : np.ndarray
        Array binario con header + pixel data.

    Ritorna
    -------
    image : np.ndarray
        Matrice 2D dell'immagine ricostruita.
    """
    # Leggi header: altezza (16 bit) + larghezza (16 bit)
    h = 0
    for i in range(16):
        h = (h << 1) | int(bits[i])
    w = 0
    for i in range(16, 32):
        w = (w << 1) | int(bits[i])

    # Leggi pixel data
    pixel_bits = bits[32:]
    n_pixels = h * w
    pixels = []
    for i in range(0, min(n_pixels * 8, len(pixel_bits)), 8):
        byte_val = 0
        for bit in pixel_bits[i:i + 8]:
            byte_val = (byte_val << 1) | int(bit)
        pixels.append(byte_val)

    # Ricostruisci matrice
    pixels = np.array(pixels[:n_pixels], dtype=np.uint8)
    if len(pixels) < n_pixels:
        pixels = np.pad(pixels, (0, n_pixels - len(pixels)))

    return pixels.reshape(h, w).astype(np.float64)


# ─────────────────────────────────────────────────────────────
#  Calcolo Capacità
# ─────────────────────────────────────────────────────────────

def compute_capacity(
    roi_matrix: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
) -> dict:
    """
    Calcola la capacità di embedding della ROI: quanti bit si possono
    nascondere dato il block size e la strategia di SV.

    Parametri
    ---------
    roi_matrix : np.ndarray
        Matrice 2D della ROI.
    block_size : int
        Dimensione dei blocchi (default: 8).
    sv_range : str
        Quali valori singolari usare: 'first', 'mid', 'last'.

    Ritorna
    -------
    info : dict
        Dizionario con: n_blocks, bits_per_block, total_bits, max_text_chars, ecc.
    """
    h, w = roi_matrix.shape[:2]

    # Numero di blocchi
    h_blocks = int(np.ceil(h / block_size))
    w_blocks = int(np.ceil(w / block_size))
    n_blocks = h_blocks * w_blocks

    # Numero di SV per blocco
    sv_per_block = block_size  # Un blocco block_size×block_size ha al massimo block_size SV

    # Bit incorporabili per blocco in base alla strategia
    bits_per_block = _get_sv_indices_count(sv_per_block, sv_range)

    total_bits = n_blocks * bits_per_block
    max_text_chars = (total_bits // 8) - 1  # -1 per il terminatore

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
    """Ritorna il numero di SV utilizzabili per un dato range."""
    return len(_get_sv_indices(n_sv, sv_range))


def _get_sv_indices(n_sv: int, sv_range: str) -> list[int]:
    """
    Ritorna gli indici dei valori singolari da modificare.

    Parametri
    ---------
    n_sv : int
        Numero totale di valori singolari nel blocco.
    sv_range : str
        'first', 'mid', o 'last'.

    Ritorna
    -------
    indices : list[int]
        Lista degli indici dei SV da alterare.
    """
    sv_range = sv_range.lower()

    if n_sv <= 2:
        # Blocchi molto piccoli: usa l'unico o i pochi SV disponibili
        return list(range(n_sv))

    if sv_range == "first":
        # Primi ~1/3 dei SV (massima robustezza, ma artefatti visibili)
        end = max(1, n_sv // 3)
        return list(range(0, end))

    elif sv_range == "last":
        # Ultimi ~1/3 dei SV (invisibile, ma vulnerabile a compressione)
        start = n_sv - max(1, n_sv // 3)
        return list(range(start, n_sv))

    elif sv_range == "mid":
        # SV intermedi: miglior compromesso
        # Salta il primo e l'ultimo terzo
        start = max(1, n_sv // 3)
        end = n_sv - max(1, n_sv // 3)
        if start >= end:
            # Fallback: usa almeno un SV nel mezzo
            mid = n_sv // 2
            return [mid]
        return list(range(start, end))

    else:
        raise ValueError(f"sv_range non valido: '{sv_range}'. Usa 'first', 'mid', o 'last'.")


# ─────────────────────────────────────────────────────────────
#  Embedding QIM
# ─────────────────────────────────────────────────────────────

def _qim_embed(sigma_val: float, bit: int, delta: float) -> float:
    """
    Incorpora un singolo bit in un valore singolare usando QIM
    (Quantization Index Modulation).

    L'idea: quantizzare σ a multipli di Δ. Se il bit è 0, si sceglie
    il multiplo pari più vicino; se il bit è 1, si sceglie il dispari.

    Parametri
    ---------
    sigma_val : float
        Il valore singolare originale.
    bit : int
        Il bit da nascondere (0 o 1).
    delta : float
        Il passo di quantizzazione. Più grande → più robusto ma più visibile.

    Ritorna
    -------
    sigma_modified : float
        Il valore singolare modificato.
    """
    # Quantizzazione: arrotonda a multiplo di delta/2
    q = np.round(sigma_val / delta)
    q = int(q)

    # Se il bit e la parità di q non corrispondono, aggiusta
    if (q % 2) != bit:
        # Scegli la direzione che minimizza la distorsione
        q_up = q + 1
        q_down = q - 1
        dist_up = abs(sigma_val - q_up * delta)
        dist_down = abs(sigma_val - q_down * delta)

        if dist_up <= dist_down:
            q = q_up
        else:
            q = q_down

    return q * delta


def _qim_extract(sigma_val: float, delta: float) -> int:
    """
    Estrae un singolo bit da un valore singolare usando QIM.

    Parametri
    ---------
    sigma_val : float
        Il valore singolare (potenzialmente modificato).
    delta : float
        Lo stesso passo di quantizzazione usato in fase di embedding.

    Ritorna
    -------
    bit : int
        Il bit estratto (0 o 1).
    """
    q = int(np.round(sigma_val / delta))
    return q % 2


# ─────────────────────────────────────────────────────────────
#  Embedding Pipeline
# ─────────────────────────────────────────────────────────────

def embed_message(
    roi_matrix: np.ndarray,
    payload_bits: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
) -> tuple[np.ndarray, dict]:
    """
    Incorpora un payload binario nella matrice della ROI modificando i
    valori singolari dei blocchi tramite SVD custom + QIM.

    Pipeline:
    1. Suddivide la ROI in blocchi (block_size × block_size)
    2. Per ogni blocco, calcola la SVD custom → U, Σ, Vᵀ
    3. Seleziona i SV da modificare (in base a sv_range)
    4. Applica QIM per incorporare i bit del payload
    5. Ricostruisce i blocchi: U · Σ_mod · Vᵀ
    6. Ricompone l'immagine

    Parametri
    ---------
    roi_matrix : np.ndarray
        Matrice 2D della ROI (float64, valori ~0-255).
    payload_bits : np.ndarray
        Array di 0 e 1 (il messaggio in formato binario).
    block_size : int
        Dimensione dei blocchi (default: 8).
    sv_range : str
        'first', 'mid', 'last' — quali SV alterare.
    delta : float
        Passo di quantizzazione QIM. Più grande → più robusto ma più visibile.
        Valori tipici: 5-30.

    Ritorna
    -------
    stego_roi : np.ndarray
        La ROI con il messaggio incorporato (stessa dimensione di roi_matrix).
    embed_info : dict
        Informazioni sull'embedding (bit usati, capacità, ecc.).
    """
    # 1. Suddividi in blocchi
    blocks, positions, original_shape = split_into_blocks(roi_matrix, block_size)

    total_payload_bits = len(payload_bits)
    bit_index = 0  # puntatore al prossimo bit da incorporare
    bits_embedded = 0

    modified_blocks = []

    for block_idx, block in enumerate(blocks):
        # 2. SVD custom sul blocco
        U_b, sigma_b, Vt_b = svd_compact(block)
        n_sv = len(sigma_b)

        # 3. Seleziona gli indici dei SV da modificare
        sv_indices = _get_sv_indices(n_sv, sv_range)

        # 4. Modifica i SV selezionati con QIM
        sigma_modified = sigma_b.copy()
        for sv_idx in sv_indices:
            if bit_index >= total_payload_bits:
                break  # Payload esaurito

            bit = int(payload_bits[bit_index])
            sigma_modified[sv_idx] = _qim_embed(sigma_b[sv_idx], bit, delta)
            bit_index += 1
            bits_embedded += 1

        # 5. Ricostruisci il blocco
        block_recon = reconstruct(U_b, sigma_modified, Vt_b)
        modified_blocks.append(block_recon)

    # 6. Ricomponi l'immagine
    stego_roi = merge_blocks(modified_blocks, positions, original_shape, block_size)

    # Informazioni sull'embedding
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
        print(f"  ⚠️  ATTENZIONE: il payload ({total_payload_bits} bit) supera la capacità "
              f"({capacity['total_bits']} bit). Sono stati incorporati solo {bits_embedded} bit.")

    return stego_roi, embed_info


# ─────────────────────────────────────────────────────────────
#  Extraction Pipeline
# ─────────────────────────────────────────────────────────────

def extract_message(
    stego_roi_matrix: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
    max_bits: int | None = None,
) -> np.ndarray:
    """
    Estrae il payload binario dalla ROI della stego-image.

    Questa è un'estrazione blind: non richiede l'immagine originale.
    Si basa sulla parità del quoziente di quantizzazione dei valori singolari.

    Parametri
    ---------
    stego_roi_matrix : np.ndarray
        Matrice della ROI dell'immagine stego.
    block_size : int
        Dimensione dei blocchi (deve corrispondere all'embedding).
    sv_range : str
        Range dei SV usati in embedding.
    delta : float
        Passo di quantizzazione (deve corrispondere all'embedding).
    max_bits : int | None
        Numero massimo di bit da estrarre. Se None, estrae tutto il possibile.

    Ritorna
    -------
    extracted_bits : np.ndarray
        Array di 0 e 1 con i bit estratti.
    """
    blocks, positions, original_shape = split_into_blocks(stego_roi_matrix, block_size)

    extracted_bits = []

    for block in blocks:
        U_b, sigma_b, Vt_b = svd_compact(block)
        n_sv = len(sigma_b)
        sv_indices = _get_sv_indices(n_sv, sv_range)

        for sv_idx in sv_indices:
            bit = _qim_extract(sigma_b[sv_idx], delta)
            extracted_bits.append(bit)

            if max_bits is not None and len(extracted_bits) >= max_bits:
                return np.array(extracted_bits, dtype=np.int32)

    return np.array(extracted_bits, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
#  Funzioni di Supporto Steganografia Completa
# ─────────────────────────────────────────────────────────────

def embed_in_full_image(
    image_matrix: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    payload_bits: np.ndarray,
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
) -> tuple[np.ndarray, dict]:
    """
    Incorpora il payload nella ROI dell'immagine completa e ritorna la stego-image.

    Parametri
    ---------
    image_matrix : np.ndarray
        Matrice 2D dell'immagine completa (grayscale).
    roi_coords : tuple[int, int, int, int]
        Coordinate della ROI: (y1, x1, y2, x2).
    payload_bits : np.ndarray
        Il messaggio in formato binario.
    block_size : int
        Dimensione dei blocchi.
    sv_range : str
        Range dei valori singolari.
    delta : float
        Passo di quantizzazione.

    Ritorna
    -------
    stego_image : np.ndarray
        L'immagine completa con il payload incorporato.
    embed_info : dict
        Informazioni sull'embedding.
    """
    y1, x1, y2, x2 = roi_coords

    # Estrai la ROI
    roi = image_matrix[y1:y2, x1:x2].copy()

    # Embedding nella ROI
    stego_roi, embed_info = embed_message(
        roi, payload_bits, block_size, sv_range, delta
    )

    # Inserisci la ROI modificata nell'immagine completa
    stego_image = image_matrix.copy()
    stego_image[y1:y2, x1:x2] = stego_roi

    # Aggiungi le coordinate ROI all'info
    embed_info["roi_coords"] = roi_coords

    return stego_image, embed_info


def extract_from_full_image(
    stego_image: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    block_size: int = 8,
    sv_range: str = "mid",
    delta: float = 15.0,
    max_bits: int | None = None,
) -> np.ndarray:
    """
    Estrae il payload dalla ROI dell'immagine stego.

    Parametri
    ---------
    stego_image : np.ndarray
        Immagine stego completa.
    roi_coords : tuple[int, int, int, int]
        Coordinate della ROI: (y1, x1, y2, x2).
    block_size : int
        Dimensione dei blocchi.
    sv_range : str
        Range dei SV usati.
    delta : float
        Passo di quantizzazione.
    max_bits : int | None
        Numero massimo di bit da estrarre.

    Ritorna
    -------
    extracted_bits : np.ndarray
        Array di bit estratti.
    """
    y1, x1, y2, x2 = roi_coords
    roi = stego_image[y1:y2, x1:x2].copy()

    return extract_message(roi, block_size, sv_range, delta, max_bits)


def print_embed_report(embed_info: dict) -> None:
    """Stampa un report leggibile dell'embedding effettuato."""
    sv_names = {
        "first": "Primi (massima robustezza, artefatti visibili)",
        "mid": "Intermedi (miglior compromesso)",
        "last": "Ultimi (invisibile, vulnerabile a JPEG)",
    }

    print(f"\n{'═' * 60}")
    print(f"  REPORT EMBEDDING")
    print(f"{'═' * 60}")
    print(f"\n  📊 Parametri:")
    print(f"     Block size:          {embed_info['block_size']}×{embed_info['block_size']}")
    print(f"     Valori singolari:    {embed_info['sv_range']} — {sv_names.get(embed_info['sv_range'], '?')}")
    print(f"     Delta (QIM):         {embed_info['delta']}")
    print(f"     Numero blocchi:      {embed_info['n_blocks']}")
    print(f"\n  📦 Payload:")
    print(f"     Bit del payload:     {embed_info['total_payload_bits']}")
    print(f"     Bit incorporati:     {embed_info['bits_embedded']}")
    print(f"     Capacità totale:     {embed_info['capacity_bits']} bit")
    print(f"     Utilizzo capacità:   {embed_info['utilization_pct']:.1f}%")

    if embed_info["payload_fully_embedded"]:
        print(f"\n  ✅ Payload incorporato con successo!")
    else:
        print(f"\n  ❌ Payload TRONCATO! Capacità insufficiente.")

    if "roi_coords" in embed_info:
        y1, x1, y2, x2 = embed_info["roi_coords"]
        print(f"     ROI: ({x1},{y1})→({x2},{y2})")

    print(f"\n{'─' * 60}")
