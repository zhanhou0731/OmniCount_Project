import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, List, Tuple, Callable

class ImageCanvas(tk.Canvas):
    """
    A custom Canvas widget for displaying images, selecting ROIs (cropping), 
    and drawing detection results.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.image_bgr: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.scale_ratio = 1.0
        self.img_draw_x = 0
        self.img_draw_y = 0
        
        # Crop Mode State
        self.crop_mode_active = False
        self.on_crop_callback: Optional[Callable[[int, int, int, int], None]] = None
        self.rect_id: Optional[int] = None
        self.start_x = 0
        self.start_y = 0
        
        # Results State
        self.current_results: List[List[int]] = []
        self.crop_coords: Optional[Tuple[int, int, int, int]] = None

        # Bindings
        self.bind("<ButtonPress-1>", self._on_mouse_press)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_release)

    def set_image(self, image_bgr: np.ndarray):
        """Sets the image to be displayed and refreshes the canvas."""
        self.image_bgr = image_bgr
        self.refresh()

    def refresh(self):
        """Resizes the image to fit the canvas and redraws everything."""
        if self.image_bgr is None:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        h, w = self.image_bgr.shape[:2]
        self.scale_ratio = min(canvas_w / w, canvas_h / h)
        
        new_w, new_h = int(w * self.scale_ratio), int(h * self.scale_ratio)
        ui_img_bgr = cv2.resize(self.image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        img_rgb = cv2.cvtColor(ui_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        
        self.delete("all")
        self.img_draw_x = (canvas_w - new_w) // 2
        self.img_draw_y = (canvas_h - new_h) // 2
        self.create_image(self.img_draw_x, self.img_draw_y, anchor=tk.NW, image=self.tk_image)
        
        self._redraw_crop_rect()
        self._redraw_results()

    def enable_crop_mode(self, callback: Callable[[int, int, int, int], None]):
        """Enables ROI selection mode."""
        self.crop_mode_active = True
        self.on_crop_callback = callback
        self.config(cursor="cross")

    def disable_crop_mode(self):
        """Disables ROI selection mode."""
        self.crop_mode_active = False
        self.config(cursor="")

    def clear_results(self):
        """Clears all detection boxes from the canvas."""
        self.delete("result_box")
        self.current_results = []

    def draw_results(self, results: List[List[int]]):
        """Draws bounding boxes for the detection results."""
        self.current_results = results
        self._redraw_results()

    def _redraw_results(self):
        self.delete("result_box")
        for box in self.current_results:
            x1, y1, x2, y2, _ = box
            ux1 = int(x1 * self.scale_ratio) + self.img_draw_x
            uy1 = int(y1 * self.scale_ratio) + self.img_draw_y
            ux2 = int(x2 * self.scale_ratio) + self.img_draw_x
            uy2 = int(y2 * self.scale_ratio) + self.img_draw_y
            self.create_rectangle(ux1, uy1, ux2, uy2, outline="#00FF00", width=2, tags="result_box")

    def _redraw_crop_rect(self):
        if self.crop_coords:
            x1, y1, x2, y2 = self.crop_coords
            ux1 = int(x1 * self.scale_ratio) + self.img_draw_x
            uy1 = int(y1 * self.scale_ratio) + self.img_draw_y
            ux2 = int(x2 * self.scale_ratio) + self.img_draw_x
            uy2 = int(y2 * self.scale_ratio) + self.img_draw_y
            self.create_rectangle(ux1, uy1, ux2, uy2, outline="red", width=2, dash=(4, 4), tags="crop_rect")

    def _on_mouse_press(self, event):
        if not self.crop_mode_active: return
        self.delete("crop_rect")
        self.start_x, self.start_y = event.x, event.y
        self.rect_id = self.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, 
                                            outline="red", width=2, dash=(4, 4), tags="crop_rect")

    def _on_mouse_drag(self, event):
        if not self.crop_mode_active or not self.rect_id: return
        self.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def _on_mouse_release(self, event):
        if not self.crop_mode_active: return
        self.disable_crop_mode()
        
        end_x, end_y = event.x, event.y
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            self.delete("crop_rect")
            return

        # Map UI coords back to real image coords
        rx1 = int((self.start_x - self.img_draw_x) / self.scale_ratio)
        ry1 = int((self.start_y - self.img_draw_y) / self.scale_ratio)
        rx2 = int((end_x - self.img_draw_x) / self.scale_ratio)
        ry2 = int((end_y - self.img_draw_y) / self.scale_ratio)
        
        self.crop_coords = (rx1, ry1, rx2, ry2)
        if self.on_crop_callback:
            self.on_crop_callback(rx1, ry1, rx2, ry2)
