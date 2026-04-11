import cv2
import numpy as np
from typing import Optional, Tuple, List, Any

class ImageProcessor:
    """Handles loading, resizing, and preprocessing of images for the OmniCount system."""
    
    def __init__(self, max_width: int = 1000):
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.max_width = max_width
        self.scale_ratio = 1.0
        self.template_image: Optional[np.ndarray] = None
        self.ui_scale_ratio = 1.0

    def load_image(self, file_path: str) -> np.ndarray:
        """Loads the image from disk and applies initial safety resizing."""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read the image file at {file_path}")
        
        # Safety Resize
        h, w = img.shape[:2]
        if w > self.max_width:
            self.scale_ratio = self.max_width / w
            new_dim = (self.max_width, int(h * self.scale_ratio))
            self.original_image = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        else:
            self.original_image = img.copy()
            self.scale_ratio = 1.0
            
        return self.original_image

    def to_original_coords(self, boxes: List[List[Any]]) -> List[List[Any]]:
        """Maps coordinates from the safety-resized image back to the original image dimensions."""
        if self.scale_ratio == 1.0:
            return boxes
        
        scaled = []
        for box in boxes:
            # box is [x1, y1, x2, y2, score]
            x1, y1, x2, y2 = box[:4]
            score = box[4] if len(box) > 4 else 0
            scaled.append([
                int(x1 / self.scale_ratio),
                int(y1 / self.scale_ratio),
                int(x2 / self.scale_ratio),
                int(y2 / self.scale_ratio),
                score
            ])
        return scaled

    def from_original_coords(self, boxes: List[List[Any]]) -> List[List[Any]]:
        """Maps coordinates from the original image dimensions to the safety-resized image dimensions."""
        if self.scale_ratio == 1.0:
            return boxes
        
        scaled = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            score = box[4] if len(box) > 4 else 0
            scaled.append([
                int(x1 * self.scale_ratio),
                int(y1 * self.scale_ratio),
                int(x2 * self.scale_ratio),
                int(y2 * self.scale_ratio),
                score
            ])
        return scaled

    @staticmethod
    def apply_preprocessing(image: np.ndarray, 
                           mode: str = "grayscale", 
                           filter_type: str = "none", 
                           use_clahe: bool = False) -> np.ndarray:
        """
        Applies preprocessing steps to a given image array.
        Can be used for both main images and templates.
        """
        processed = image.copy()

        # 1. Noise Filters
        if filter_type == "gaussian":
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
        elif filter_type == "bilateral":
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

        # 2. CLAHE Enhancement
        if use_clahe:
            if len(processed.shape) == 2:  # Grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
            else:  # Color (LAB space)
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                lab = cv2.merge((cl, a, b))
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Image Mode (Grayscale, Color, or Edges)
        if mode in ["grayscale", "edges"]:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            if mode == "edges":
                processed = cv2.Canny(processed, 50, 150)

        return processed

    def get_processed_main_and_template(self, 
                                       mode: str = "grayscale", 
                                       filter_type: str = "none", 
                                       use_clahe: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses both the main image and the template image."""
        if self.original_image is None or self.template_image is None:
            raise ValueError("Main image and template must be loaded/cropped before preprocessing.")

        processed_main = self.apply_preprocessing(self.original_image, mode, filter_type, use_clahe)
        processed_temp = self.apply_preprocessing(self.template_image, mode, filter_type, use_clahe)
        
        return processed_main, processed_temp

    def get_ui_image(self, canvas_width: int, canvas_height: int) -> Optional[np.ndarray]:
        if self.original_image is None:
            return None
        
        h, w = self.original_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        
        new_w, new_h = int(w * scale), int(h * scale)
        ui_image = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        self.ui_scale_ratio = scale 
        return ui_image

    def crop_template(self, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        if self.original_image is None:
            return None

        start_x, end_x = sorted([x1, x2])
        start_y, end_y = sorted([y1, y2])

        h, w = self.original_image.shape[:2]
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w, end_x), min(h, end_y)

        self.template_image = self.original_image[start_y:end_y, start_x:end_x]
        return self.template_image
