import numpy as np

class BoundingBox:

    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 confidence: float, class_id: int, class_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return (f"BoundingBox({self.class_name}, "
                f"({self.x1},{self.y1})->({self.x2},{self.y2}), "
                f"area={self.area}, conf={self.confidence:.2f})")

class ROIResult:

    def __init__(
        self,
        mask: np.ndarray,
        bounding_boxes: list[BoundingBox],
        strategy: str,
        selected_box: BoundingBox | None = None,
    ):

        self.mask = mask
        self.bounding_boxes = bounding_boxes
        self.strategy = strategy
        self.selected_box = selected_box

    @property
    def roi_pixel_count(self) -> int:
        return int(np.sum(self.mask))


def select_roi(
    image_shape: tuple[int, int],
    bounding_boxes: list[BoundingBox],
    strategy: str = "C",
    box_index: int | None = None,
) -> ROIResult:

    h, w = image_shape
    strategy = strategy.upper()

    if strategy not in ("A", "B", "C"):
        raise ValueError(f"Strategia non valida: '{strategy}'. Usa 'A', 'B', o 'C'.")

    if len(bounding_boxes) == 0:

        print("Nessun oggetto rilevato da YOLO. Si usa l'intera immagine come ROI.")
        mask = np.ones((h, w), dtype=bool)
        return ROIResult(mask=mask, bounding_boxes=[], strategy=strategy, selected_box=None)

    if strategy == "A":

        if box_index is not None:
            if box_index < 0 or box_index >= len(bounding_boxes):
                raise ValueError(
                    f"box_index={box_index} fuori range. "
                    f"Disponibili: 0-{len(bounding_boxes) - 1}"
                )
            selected = bounding_boxes[box_index]
        else:
            selected = bounding_boxes[0]

        mask = np.zeros((h, w), dtype=bool)
        mask[selected.y1:selected.y2, selected.x1:selected.x2] = True
        return ROIResult(
            mask=mask, bounding_boxes=bounding_boxes,
            strategy="A", selected_box=selected,
        )

    elif strategy == "B":

        mask = np.ones((h, w), dtype=bool)
        for bb in bounding_boxes:
            mask[bb.y1:bb.y2, bb.x1:bb.x2] = False
        return ROIResult(
            mask=mask, bounding_boxes=bounding_boxes,
            strategy="B", selected_box=None,
        )

    else:

        largest = bounding_boxes[0]
        mask = np.zeros((h, w), dtype=bool)
        mask[largest.y1:largest.y2, largest.x1:largest.x2] = True
        return ROIResult(
            mask=mask, bounding_boxes=bounding_boxes,
            strategy="C", selected_box=largest,
        )

def extract_roi_region(
    image_matrix: np.ndarray,
    roi_result: ROIResult,
) -> np.ndarray:

    if roi_result.selected_box is not None:
        bb = roi_result.selected_box
        return image_matrix[bb.y1:bb.y2, bb.x1:bb.x2].copy()
    else:

        masked = image_matrix.copy()
        if masked.ndim == 2:
            masked[~roi_result.mask] = 0
        else:
            masked[~roi_result.mask] = 0
        return masked
