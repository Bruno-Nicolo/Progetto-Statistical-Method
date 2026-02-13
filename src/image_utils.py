"""
Utilità per la gestione delle immagini nel contesto della SVD steganography.

Funzioni per:
- Caricare e convertire immagini in matrici numpy
- Applicare e rimuovere il mean centering
- Ricostruire immagini da matrici
- Suddividere e ricomporre immagini in blocchi
"""

import numpy as np
from PIL import Image


def load_image_as_matrix(image_path: str, grayscale: bool = True) -> np.ndarray:
    """
    Carica un'immagine e la converte in una matrice numpy bidimensionale.

    Parametri
    ---------
    image_path : str
        Percorso del file immagine.
    grayscale : bool
        Se True, converte l'immagine in scala di grigi (matrice 2D).
        Se False, ritorna una matrice 3D (H×W×C).

    Ritorna
    -------
    matrix : np.ndarray
        Matrice numpy dell'immagine (float64, valori 0-255).
    """
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')  # Scala di grigi
    else:
        img = img.convert('RGB')

    return np.array(img, dtype=np.float64)


def matrix_to_image(matrix: np.ndarray) -> Image.Image:
    """
    Converte una matrice numpy in un oggetto PIL Image.

    I valori vengono clippati a [0, 255] e convertiti a uint8.

    Parametri
    ---------
    matrix : np.ndarray
        Matrice 2D (grayscale) o 3D (RGB).

    Ritorna
    -------
    img : PIL.Image.Image
        L'immagine risultante.
    """
    # Clip e conversione a uint8
    clipped = np.clip(matrix, 0, 255).astype(np.uint8)

    if clipped.ndim == 2:
        return Image.fromarray(clipped, mode='L')
    elif clipped.ndim == 3:
        return Image.fromarray(clipped, mode='RGB')
    else:
        raise ValueError(f"Dimensione matrice non supportata: {clipped.ndim}D")


def save_image(matrix: np.ndarray, output_path: str) -> None:
    """
    Salva una matrice numpy come file immagine.

    Parametri
    ---------
    matrix : np.ndarray
        Matrice dell'immagine.
    output_path : str
        Percorso di output per il file immagine.
    """
    img = matrix_to_image(matrix)
    img.save(output_path)


def apply_mean_centering(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applica il mean centering alla matrice: sottrae la media di ogni colonna (feature).

    Il mean centering è essenziale per la PCA tradizionale e assicura di catturare
    la struttura intrinseca dei dati. Tuttavia, nella compressione di immagini
    grezze potrebbe non essere necessario.

    Parametri
    ---------
    X : np.ndarray
        Matrice di input (m×n).

    Ritorna
    -------
    X_centered : np.ndarray
        Matrice con media sottratta.
    means : np.ndarray
        Vettore delle medie (una per colonna), necessario per la ricostruzione.
    """
    means = np.mean(X, axis=0)
    X_centered = X - means
    return X_centered, means


def remove_mean_centering(X_centered: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Rimuove il mean centering, riagiungendo la media a ogni colonna.

    Parametri
    ---------
    X_centered : np.ndarray
        Matrice centrata.
    means : np.ndarray
        Vettore delle medie (ottenuto da apply_mean_centering).

    Ritorna
    -------
    X_original : np.ndarray
        Matrice con i valori originali ripristinati.
    """
    return X_centered + means


def split_into_blocks(matrix: np.ndarray, block_size: int = 8) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Suddivide una matrice in blocchi di dimensione block_size × block_size.

    Se le dimensioni dell'immagine non sono multipli esatti di block_size,
    la matrice viene estesa con padding (zero-padding).

    Parametri
    ---------
    matrix : np.ndarray
        Matrice 2D dell'immagine.
    block_size : int
        Dimensione del lato dei blocchi (default: 8).

    Ritorna
    -------
    blocks : list[np.ndarray]
        Lista dei blocchi estratti.
    positions : list[tuple[int, int]]
        Posizioni (riga, colonna) di ogni blocco nella matrice paddingata.
    original_shape : tuple[int, int]
        Dimensioni originali della matrice (prima del padding).
    """
    original_shape = matrix.shape[:2]
    h, w = original_shape

    # Calcola le dimensioni con padding
    h_padded = int(np.ceil(h / block_size)) * block_size
    w_padded = int(np.ceil(w / block_size)) * block_size

    # Applica il padding
    padded = np.zeros((h_padded, w_padded), dtype=matrix.dtype)
    padded[:h, :w] = matrix

    blocks = []
    positions = []

    for i in range(0, h_padded, block_size):
        for j in range(0, w_padded, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            blocks.append(block.copy())
            positions.append((i, j))

    return blocks, positions, original_shape


def merge_blocks(blocks: list[np.ndarray], positions: list[tuple[int, int]], original_shape: tuple[int, int], block_size: int = 8) -> np.ndarray:
    """
    Ricompone una matrice a partire da una lista di blocchi.

    Parametri
    ---------
    blocks : list[np.ndarray]
        Lista dei blocchi.
    positions : list[tuple[int, int]]
        Posizioni (riga, colonna) di ogni blocco.
    original_shape : tuple[int, int]
        Dimensioni originali della matrice (prima del padding).
    block_size : int
        Dimensione del lato dei blocchi.

    Ritorna
    -------
    matrix : np.ndarray
        Matrice ricostruita con le dimensioni originali.
    """
    h, w = original_shape
    h_padded = int(np.ceil(h / block_size)) * block_size
    w_padded = int(np.ceil(w / block_size)) * block_size

    result = np.zeros((h_padded, w_padded), dtype=np.float64)

    for block, (i, j) in zip(blocks, positions):
        result[i:i + block_size, j:j + block_size] = block

    # Rimuove il padding
    return result[:h, :w]
