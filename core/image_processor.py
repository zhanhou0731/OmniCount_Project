import cv2
import numpy as np

class ImageProcessor:
    """Handles loading, resizing, and preprocessing of images for the OmniCount system."""
    
    def __init__(self, max_width=1000):
        self.original_image = None
        self.processed_image = None
        self.max_width = max_width
        self.scale_ratio = 1.0
        self.template_image = None

    def load_image(self, file_path):
        """Loads the image from disk and applies initial safety resizing."""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read the image file.")
        
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

    def preprocess(self, mode="grayscale", filter_type="bilateral"):
        if self.original_image is None:
            return None

        # 1. Color Conversion
        if mode == "grayscale":
            processed = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            processed = self.original_image.copy()

        # 2. User-Selected Noise Reduction
        if filter_type == "gaussian":
            # ksize=(5,5) is a standard moderate blur, sigmaX=0 lets OpenCV calculate it
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
            
        elif filter_type == "bilateral":
            # d=9, sigmaColor=75, sigmaSpace=75 (Blurs textures but keeps edges sharp)
            if len(processed.shape) == 2: # If Grayscale
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
            else: # If Color
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
                
        elif filter_type == "none":
            pass # Skip blurring completely

        self.processed_image = processed
        return self.processed_image



    def get_ui_image(self, canvas_width, canvas_height):
        if self.original_image is None:
            return None
        
        # Get current dimensions
        h, w = self.original_image.shape[:2]
        
        # Calculate the scaling factor to fit the canvas
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        
        # Pick the SMALLEST scale factor so the whole image fits inside the box
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize specifically for the UI
        ui_image = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        self.ui_scale_ratio = scale 
        
        return ui_image

    
    def crop_template(self, x1, y1, x2, y2):
        if self.original_image is None:
            return None

        start_x, end_x = min(x1, x2), max(x1, x2)
        start_y, end_y = min(y1, y2), max(y1, y2)

        h, w = self.original_image.shape[:2]
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w, end_x), min(h, end_y)

        self.template_image = self.original_image[start_y:end_y, start_x:end_x]
        return self.template_image