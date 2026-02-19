import numpy as np
from PIL import Image

def load_image_as_matrix(image_path: str, grayscale: bool = True) -> np.ndarray:

    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    return np.array(img, dtype=np.float64)

def matrix_to_image(matrix: np.ndarray) -> Image.Image:

    clipped = np.clip(matrix, 0, 255).astype(np.uint8)

    if clipped.ndim == 2:
        return Image.fromarray(clipped, mode='L')
    elif clipped.ndim == 3:
        return Image.fromarray(clipped, mode='RGB')
    else:
        raise ValueError(f"Dimensione matrice non supportata: {clipped.ndim}D")

def save_image(matrix: np.ndarray, output_path: str) -> None:

    img = matrix_to_image(matrix)
    img.save(output_path)



def split_into_blocks(matrix: np.ndarray, block_size: int = 8) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:

    original_shape = matrix.shape[:2]
    h, w = original_shape

    h_padded = int(np.ceil(h / block_size)) * block_size
    w_padded = int(np.ceil(w / block_size)) * block_size

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

    h, w = original_shape
    h_padded = int(np.ceil(h / block_size)) * block_size
    w_padded = int(np.ceil(w / block_size)) * block_size

    result = np.zeros((h_padded, w_padded), dtype=np.float64)

    for block, (i, j) in zip(blocks, positions):
        result[i:i + block_size, j:j + block_size] = block

    return result[:h, :w]

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)