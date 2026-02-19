import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

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

def load_yolo_model(model_name: str = "yolov8n.pt") -> "YOLO":

    if YOLO is None:
        raise ImportError(
            "La libreria 'ultralytics' non è installata.\n"
            "Installala con: pip install ultralytics\n"
            "Oppure: pip install r requirements.txt"
        )
    print("Caricamento modello YOLO: {model_name}...")
    model = YOLO(model_name)
    print("Modello caricato!")
    return model

def detect_objects(
    model: "YOLO",
    image_path: str,
    confidence_threshold: float = 0.25,
) -> list[BoundingBox]:

    results = model(image_path, verbose=False, conf=confidence_threshold)

    if not results or len(results) == 0:
        return []

    result = results[0]
    boxes_data = result.boxes

    if boxes_data is None or len(boxes_data) == 0:
        return []

    bounding_boxes = []
    names = result.names

    for i in range(len(boxes_data)):

        xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        conf = float(boxes_data.conf[i].cpu().numpy())
        cls_id = int(boxes_data.cls[i].cpu().numpy())
        cls_name = names.get(cls_id, f"classe_{cls_id}")

        bb = BoundingBox(x1, y1, x2, y2, conf, cls_id, cls_name)
        bounding_boxes.append(bb)

    bounding_boxes.sort(key=lambda b: b.area, reverse=True)
    return bounding_boxes

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

def draw_detections(
    image_path: str,
    bounding_boxes: list[BoundingBox],
    roi_result: ROIResult | None = None,
) -> Image.Image:

    from PIL import ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for bb in bounding_boxes:
        color = "lime"
        if roi_result and roi_result.selected_box == bb:
            color = "red"

        draw.rectangle(
            [bb.x1, bb.y1, bb.x2, bb.y2],
            outline=color,
            width=3,
        )
        label = f"{bb.class_name} ({bb.confidence:.0%})"

        text_y = max(bb.y1 - 18, 0)
        draw.text((bb.x1 + 2, text_y), label, fill=color)

    if roi_result and roi_result.strategy == "B":
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for bb in bounding_boxes:
            overlay_draw.rectangle(
                [bb.x1, bb.y1, bb.x2, bb.y2],
                fill=(255, 0, 0, 60),
            )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img

def print_detection_report(
    bounding_boxes: list[BoundingBox],
    roi_result: ROIResult,
    image_shape: tuple[int, int],
) -> None:

    h, w = image_shape
    total_pixels = h * w

    strategy_names = {
        "A": "Soggetto (bounding box specifico)",
        "B": "Sfondo (escludendo tutti i bounding box)",
        "C": "Automatico (bounding box più grande)",
    }

    print(f"{'' * 60}")
    print("RILEVAMENTO OGGETTI — YOLOv8")
    print(f"{'' * 60}")
    print("Dimensioni immagine: {h}×{w} ({total_pixels:,} pixel)")
    print(f"Oggetti rilevati: {len(bounding_boxes)}")

    if bounding_boxes:
        print(f"{'#':<4} {'Classe':<20} {'Confidenza':<12} {'Dimensioni':<15} {'Area (px)':<12}")
        print(f"{'' * 65}")
        for i, bb in enumerate(bounding_boxes):
            print(
                f"  {i:<4} {bb.class_name:<20} {bb.confidence:<12.1%} "
                f"{bb.width}×{bb.height:<10} {bb.area:<12,}"
            )

    print(f"Strategia selezionata: {roi_result.strategy} — {strategy_names[roi_result.strategy]}")

    if roi_result.selected_box:
        bb = roi_result.selected_box
        print("Bounding box scelto: {bb.class_name} "
              f"({bb.x1},{bb.y1})→({bb.x2},{bb.y2}), {bb.area:,} pixel")

    roi_pct = roi_result.roi_pixel_count / total_pixels * 100
    print("Pixel nella ROI: {roi_result.roi_pixel_count:,} ({roi_pct:.1f}% dell'immagine)")
    print(f"{'' * 60}")