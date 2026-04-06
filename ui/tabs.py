import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from core.image_processor import ImageProcessor
import cv2

class MainCounterTab(ttk.Frame):
    """Handles the UI for the Main Counting screen."""
    def __init__(self, parent):
        super().__init__(parent)
        self.processor = ImageProcessor(max_width=1200)
        self.tk_image = None 
        self.current_image_path = None
        self.resize_timer = None
        self.setup_ui()
        self.crop_mode_active = False
        self.rect_id = None 
        self.start_x = self.start_y = 0
        self.img_draw_x = self.img_draw_y = 0
        self.current_crop_coords = None

    def setup_ui(self):
        # --- Left Panel (Scrollable Container) ---
        left_container = ttk.Frame(self) # Removed strict width restrictions
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        self.control_canvas = tk.Canvas(left_container, highlightthickness=0, width=260)
        
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.control_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        self.control_frame = ttk.Frame(self.control_canvas, padding=10)
        self.control_frame_id = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        self.control_frame.bind("<Configure>", self.on_frame_configure)
        self.control_canvas.bind("<Configure>", self.on_canvas_configure)
        self.control_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        control_frame = self.control_frame
        '''
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)

        self.control_canvas = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.control_canvas.yview)

        control_frame = ttk.Frame(self.control_canvas, padding=10)

        control_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )
        self.control_canvas.create_window((0, 0), window=control_frame, anchor="nw", width=250)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        '''
        ttk.Label(control_frame, text="Setup", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="1. Upload Image", command=self.upload_image).pack(fill=tk.X, pady=2)
        self.btn_select_template = ttk.Button(control_frame, text="2. Select Template", command=self.toggle_crop_mode)
        self.btn_select_template.pack(fill=tk.X, pady=2)
        
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Parameters", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Image Mode Radio Buttons
        self.img_mode_var = tk.StringVar(value="grayscale")
        ttk.Radiobutton(control_frame, text="Grayscale (Fast)", variable=self.img_mode_var, value="grayscale").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Color (RGB)", variable=self.img_mode_var, value="color").pack(anchor=tk.W)
        
        # Filter Selection Dropdown
        ttk.Label(control_frame, text="Noise Filter:").pack(anchor=tk.W, pady=(10, 0))
        self.filter_var = tk.StringVar(value="bilateral")
        filter_dropdown = ttk.Combobox(control_frame, textvariable=self.filter_var, state="readonly")
        filter_dropdown['values'] = ("none", "gaussian", "bilateral")
        filter_dropdown.pack(fill=tk.X, pady=(0, 5))
        filter_dropdown.unbind_class("TCombobox", "<MouseWheel>")

        # Enhancements Checkboxes
        ttk.Label(control_frame, text="Enhancements:").pack(anchor=tk.W, pady=(10, 0))
        
        self.multi_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Multi-Scale Matching", variable=self.multi_scale_var).pack(anchor=tk.W)

        # Rotation Invariance Checkbox
        self.rotation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Rotation Invariance", variable=self.rotation_var).pack(anchor=tk.W, pady=(0, 5))

        # Sliders
        ttk.Label(control_frame, text="NCC Confidence Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.ncc_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, value=0.8, orient=tk.HORIZONTAL)
        self.ncc_slider.pack(fill=tk.X)

        ttk.Label(control_frame, text="NMS Overlap Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.nms_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, value=0.3, orient=tk.HORIZONTAL)
        self.nms_slider.pack(fill=tk.X)

        ttk.Label(control_frame, text="Template Preview:", font=("Arial", 9)).pack(anchor=tk.W, pady=(10, 0))
        self.template_canvas = tk.Canvas(control_frame, width=100, height=100, bg="#333333", highlightthickness=1)
        self.template_canvas.pack(pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Execute", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="START COUNTING").pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Reset Current").pack(fill=tk.X)

        # --- Right Panel ---
        canvas_frame = ttk.Frame(self, padding=10)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.create_text(300, 250, text="Upload an image to begin...", fill="white", font=("Arial", 14))
        #mouse binding
        self.main_canvas.bind("<Configure>", self.on_canvas_resize)
        self.main_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.main_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image to Count",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not file_path:
            return

        try:
            self.processor.load_image(file_path)
            self.current_crop_coords = None
            self.current_image_path = file_path
            self.refresh_canvas()
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to load image:\n{str(e)}")


    def refresh_canvas(self):
        if self.processor.original_image is None:
            return

        canvas_w = self.main_canvas.winfo_width()
        canvas_h = self.main_canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_bgr = self.processor.get_ui_image(canvas_w, canvas_h)
        ui_h, ui_w = img_bgr.shape[:2]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        
        self.main_canvas.delete("all")
        
        self.img_draw_x = (canvas_w - ui_w) // 2
        self.img_draw_y = (canvas_h - ui_h) // 2
        
        self.main_canvas.create_image(self.img_draw_x, self.img_draw_y, anchor=tk.NW, image=self.tk_image)
        
        if self.current_crop_coords:
            rx1, ry1, rx2, ry2 = self.current_crop_coords
            ratio = self.processor.ui_scale_ratio
            
            ui_x1 = int(rx1 * ratio) + self.img_draw_x
            ui_y1 = int(ry1 * ratio) + self.img_draw_y
            ui_x2 = int(rx2 * ratio) + self.img_draw_x
            ui_y2 = int(ry2 * ratio) + self.img_draw_y
            
            self.main_canvas.create_rectangle(
                ui_x1, ui_y1, ui_x2, ui_y2, 
                outline="red", width=2, dash=(4, 4), tags="crop_rect"
            )
        
    def on_canvas_resize(self, event):
        if self.processor.original_image is not None:
            if self.resize_timer is not None:
                self.after_cancel(self.resize_timer)
            
            self.resize_timer = self.after(150, self.refresh_canvas)

    
    def toggle_crop_mode(self):
        if self.processor.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
            
        self.crop_mode_active = not getattr(self, 'crop_mode_active', False)
        
        if self.crop_mode_active:
            self.btn_select_template.config(text="🔴 Cancel Selection")
            self.main_canvas.config(cursor="cross")
            print("Crop mode ON.")
        else:
            self.btn_select_template.config(text="2. Select Template")
            self.main_canvas.config(cursor="")
            print("Crop mode OFF.")

    def on_mouse_press(self, event):
        if not self.crop_mode_active:
            return
            
        self.main_canvas.delete("crop_rect") 
        
        self.start_x = event.x
        self.start_y = event.y
        
        self.rect_id = self.main_canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline="red", width=2, dash=(4, 4), tags="crop_rect"
        )

    def on_mouse_drag(self, event):
        if not self.crop_mode_active or not self.rect_id:
            return
        self.main_canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_release(self, event):
        if not self.crop_mode_active:
            return
            
        self.toggle_crop_mode()
        
        end_x, end_y = event.x, event.y
        
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            self.main_canvas.delete("crop_rect")
            return

        ratio = self.processor.ui_scale_ratio
        real_start_x = int((self.start_x - self.img_draw_x) / ratio)
        real_start_y = int((self.start_y - self.img_draw_y) / ratio)
        real_end_x = int((end_x - self.img_draw_x) / ratio)
        real_end_y = int((end_y - self.img_draw_y) / ratio)
        self.current_crop_coords = (real_start_x, real_start_y, real_end_x, real_end_y)

        template = self.processor.crop_template(real_start_x, real_start_y, real_end_x, real_end_y)
        
        if template is not None:
            print(f"Template successfully cropped! Size: {template.shape}")
            self.update_template_preview()


    def update_template_preview(self):

        temp_bgr = self.processor.template_image
        if temp_bgr is None:
            return
            
        # Convert OpenCV BGR to RGB format
        temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(temp_rgb)
        
        # Shrink the image so it fits inside the 100x100 preview box
        # .thumbnail() is smart—it keeps the aspect ratio so it doesn't squish!
        pil_img.thumbnail((100, 100)) 
        
        # Save a reference so Tkinter's garbage collector doesn't delete it
        self.tk_template_image = ImageTk.PhotoImage(pil_img) 
        
        # Draw it in the center of the preview canvas (which is 100x100)
        self.template_canvas.delete("all")
        self.template_canvas.create_image(50, 50, anchor=tk.CENTER, image=self.tk_template_image)


    def on_frame_configure(self, event):
        """Reset the scroll region to perfectly wrap the buttons."""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """When the canvas resizes, stretch the inner frame to match its width."""
        self.control_canvas.itemconfig(self.control_frame_id, width=event.width)

    def on_mousewheel(self, event):
        """Allows scrolling ONLY if the buttons are taller than the screen."""
        frame_height = self.control_frame.winfo_reqheight()
        canvas_height = self.control_canvas.winfo_height()
        
        if frame_height > canvas_height:
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class HistoryTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        action_frame = ttk.Frame(self, padding=10)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(action_frame, text="Export Selected to PDF").pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear History").pack(side=tk.RIGHT, padx=5)

        columns = ("id", "time", "image", "mode", "conf", "nms", "count")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        
        headings = ["ID", "Time", "Image Name", "Mode", "NCC Conf.", "NMS Overlap", "Total Count"]
        for col, heading in zip(columns, headings):
            self.tree.heading(col, text=heading)
            self.tree.column(col, anchor=tk.CENTER, width=100)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)