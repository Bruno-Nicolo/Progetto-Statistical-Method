import numpy as np

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

def load_yolo_model(model_path: str = "yolov8n.pt"):
    if YOLO is None:
        raise ImportError("Libreria 'ultralytics' non installata. Esegui 'pip install ultralytics'.")
    return YOLO(model_path)

def detect_objects(model, image_matrix: np.ndarray, conf: float = 0.25) -> list[BoundingBox]:
    if image_matrix.ndim == 2:
        img_3c = np.stack([image_matrix]*3, axis=-1)
    else:
        img_3c = image_matrix

    img_uint8 = np.clip(img_3c, 0, 255).astype(np.uint8)
    results = model(img_uint8, conf=conf, verbose=False)
    
    bounding_boxes = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            names = result.names
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                box_conf = float(confs[i])
                class_id = int(cls_ids[i])
                class_name = names[class_id] if names else str(class_id)
                
                bb = BoundingBox(x1, y1, x2, y2, box_conf, class_id, class_name)
                bounding_boxes.append(bb)
                
    bounding_boxes.sort(key=lambda b: b.area, reverse=True)
    return bounding_boxes

def draw_detections(image_matrix: np.ndarray, bounding_boxes: list[BoundingBox], output_path: str):
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        if image_matrix.ndim == 2:
            img_pil = Image.fromarray(np.clip(image_matrix, 0, 255).astype(np.uint8)).convert("RGB")
        else:
            img_pil = Image.fromarray(np.clip(image_matrix, 0, 255).astype(np.uint8))
            
        draw = ImageDraw.Draw(img_pil)
        
        for bb in bounding_boxes:
            # Draw rectangle
            draw.rectangle([bb.x1, bb.y1, bb.x2, bb.y2], outline="red", width=2)
            # Draw label
            label = f"{bb.class_name} ({bb.confidence:.2f})"
            draw.text((bb.x1, max(0, bb.y1 - 15)), label, fill="red")
            
        img_pil.save(output_path)
        print(f"Rilevamenti salvati in: {output_path}")
    except Exception as e:
        print(f"Errore durante il disegno dei bounding box: {e}")


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
