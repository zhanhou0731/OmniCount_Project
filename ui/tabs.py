import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from core.image_processor import ImageProcessor
from core.matcher import MatcherEngine
import cv2

class MainCounterTab(ttk.Frame):
    def __init__(self, parent, count_var):
        super().__init__(parent)
        self.count_var = count_var
        self.processor = ImageProcessor(max_width=1200)
        self.engine = MatcherEngine()
        self.tk_image = None 
        self.current_image_path = None
        self.resize_timer = None

        self.ncc_name = "Normalized Cross-Correlation (NCC)"
        self.ssd_name = "Sum of Squared Differences (SSD)"

        # UI State Variables
        self.algo_var = tk.StringVar(value=self.ncc_name)
        self.img_mode_var = tk.StringVar(value="grayscale")
        self.filter_var = tk.StringVar(value="bilateral")
        self.clahe_var = tk.BooleanVar(value=False)
        self.multi_scale_var = tk.BooleanVar(value=True)
        self.rotation_var = tk.BooleanVar(value=False)
        
        self.setup_ui()
        self.crop_mode_active = False
        self.rect_id = None 
        self.start_x = self.start_y = 0
        self.img_draw_x = self.img_draw_y = 0
        self.current_crop_coords = None
        self.current_results = []

    def setup_ui(self):
        # --- Left Panel (Scrollable Container) ---
        left_container = ttk.Frame(self)
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
        
        # --- SETUP SECTION ---
        ttk.Label(control_frame, text="Setup", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="1. Upload Image", command=self.upload_image).pack(fill=tk.X, pady=2)
        self.btn_select_template = ttk.Button(control_frame, text="2. Select Template", command=self.toggle_crop_mode)
        self.btn_select_template.pack(fill=tk.X, pady=2)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # --- PARAMETERS SECTION ---
        ttk.Label(control_frame, text="Parameters", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Algorithm Dropdown
        ttk.Label(control_frame, text="Algorithm:").pack(anchor=tk.W)
        self.algo_dropdown = ttk.Combobox(control_frame, textvariable=self.algo_var, state="readonly")
        self.algo_dropdown['values'] = (self.ncc_name, self.ssd_name)
        self.algo_dropdown.pack(fill=tk.X, pady=(0, 5))

        # Image Mode
        ttk.Label(control_frame, text="Image Mode:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Radiobutton(control_frame, text="Grayscale (Fast)", variable=self.img_mode_var, value="grayscale").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Color (RGB)", variable=self.img_mode_var, value="color").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Edges", variable=self.img_mode_var, value="edges").pack(anchor=tk.W)

        # Filters & Enhancements
        ttk.Label(control_frame, text="Noise Filter:").pack(anchor=tk.W, pady=(5, 0))
        filter_dropdown = ttk.Combobox(control_frame, textvariable=self.filter_var, state="readonly")
        filter_dropdown['values'] = ("none", "gaussian", "bilateral")
        filter_dropdown.pack(fill=tk.X, pady=(0, 5))
        filter_dropdown.unbind_class("TCombobox", "<MouseWheel>")

        ttk.Label(control_frame, text="Enhancements:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Checkbutton(control_frame, text="CLAHE", variable=self.clahe_var).pack(anchor=tk.W, pady=(5, 5))
        ttk.Checkbutton(control_frame, text="Multi-Scale Matching", variable=self.multi_scale_var).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Rotation Invariance", variable=self.rotation_var).pack(anchor=tk.W, pady=(0, 5))

        # Sliders
        self.conf_lbl_var = tk.StringVar(value="Confidence Threshold: 0.80")
        ttk.Label(control_frame, textvariable=self.conf_lbl_var).pack(anchor=tk.W, pady=(5, 0))
        self.conf_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, value=0.8, orient=tk.HORIZONTAL,
                                    command=lambda v: self.conf_lbl_var.set(f"Confidence Threshold: {float(v):.2f}"))
        self.conf_slider.pack(fill=tk.X)

        self.nms_lbl_var = tk.StringVar(value="NMS Overlap Threshold: 0.30")
        ttk.Label(control_frame, textvariable=self.nms_lbl_var).pack(anchor=tk.W, pady=(10, 0))
        self.nms_slider = ttk.Scale(control_frame, from_=0.1, to=1.0, value=0.3, orient=tk.HORIZONTAL,
                                    command=lambda v: self.nms_lbl_var.set(f"NMS Overlap Threshold: {float(v):.2f}"))
        self.nms_slider.pack(fill=tk.X)

        # Previews & Execution
        ttk.Label(control_frame, text="Template Preview:", font=("Arial", 9)).pack(anchor=tk.W, pady=(10, 0))
        self.template_canvas = tk.Canvas(control_frame, width=100, height=100, bg="#333333", highlightthickness=1)
        self.template_canvas.pack(pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Execute", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="START COUNTING", command=self.execute_counting).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Clear Results", command=self.clear_results).pack(fill=tk.X)

        # --- Right Panel ---
        canvas_frame = ttk.Frame(self, padding=10)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.create_text(300, 250, text="Upload an image to begin...", fill="white", font=("Arial", 14))
        
        self.main_canvas.bind("<Configure>", self.on_canvas_resize)
        self.main_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.main_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image to Count",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not file_path: return
        try:
            self.processor.load_image(file_path)
            self.current_crop_coords = None
            self.current_image_path = file_path
            self.refresh_canvas()
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to load image:\n{str(e)}")

    def refresh_canvas(self):
        if self.processor.original_image is None: return
        canvas_w = self.main_canvas.winfo_width()
        canvas_h = self.main_canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: return

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
            self.main_canvas.create_rectangle(ui_x1, ui_y1, ui_x2, ui_y2, outline="red", width=2, dash=(4, 4), tags="crop_rect")

        if getattr(self, 'current_results', []):
            self.draw_results(self.current_results, is_refresh=True)
        
    def on_canvas_resize(self, event):
        if self.processor.original_image is not None:
            if self.resize_timer is not None: self.after_cancel(self.resize_timer)
            self.resize_timer = self.after(150, self.refresh_canvas)
    
    def toggle_crop_mode(self):
        self.clear_results()
        if self.processor.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
            
        self.crop_mode_active = not getattr(self, 'crop_mode_active', False)
        if self.crop_mode_active:
            self.btn_select_template.config(text="🔴 Cancel Selection")
            self.main_canvas.config(cursor="cross")
        else:
            self.btn_select_template.config(text="2. Select Template")
            self.main_canvas.config(cursor="")

    def on_mouse_press(self, event):
        if not self.crop_mode_active: return
        self.main_canvas.delete("crop_rect") 
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.main_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2, dash=(4, 4), tags="crop_rect")

    def on_mouse_drag(self, event):
        if not self.crop_mode_active or not self.rect_id: return
        self.main_canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_release(self, event):
        if not self.crop_mode_active: return
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
            self.update_template_preview()

    def update_template_preview(self):
        temp_bgr = self.processor.template_image
        if temp_bgr is None: return
        temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(temp_rgb)
        pil_img.thumbnail((100, 100)) 
        self.tk_template_image = ImageTk.PhotoImage(pil_img) 
        self.template_canvas.delete("all")
        self.template_canvas.create_image(50, 50, anchor=tk.CENTER, image=self.tk_template_image)

    def on_frame_configure(self, event):
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.control_canvas.itemconfig(self.control_frame_id, width=event.width)

    def on_mousewheel(self, event):
        frame_height = self.control_frame.winfo_reqheight()
        canvas_height = self.control_canvas.winfo_height()
        if frame_height > canvas_height:
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def clear_results(self):
        self.main_canvas.delete("result_box")
        self.current_results = []
        self.count_var.set("TOTAL COUNT: 0")

    def execute_counting(self):
        if self.processor.original_image is None or self.processor.template_image is None:
            messagebox.showwarning("Warning", "Please upload an image and select a template first!")
            return

        algo = self.algo_var.get()
        img_mode = self.img_mode_var.get()
        filter_type = self.filter_var.get()
        use_clahe = self.clahe_var.get()
        use_ms = self.multi_scale_var.get()
        use_rot = self.rotation_var.get()
        conf_thresh = float(self.conf_slider.get())
        nms_thresh = float(self.nms_slider.get())

        print(f"Executing: {algo} | Mode: {img_mode} | CLAHE: {use_clahe} | Rot: {use_rot} | MS: {use_ms}")

        results = self.engine.match(
            image=self.processor.original_image,
            template=self.processor.template_image,
            algorithm=algo,
            image_mode=img_mode,
            filter_type=filter_type,
            use_clahe=use_clahe,
            nms_thresh=nms_thresh,
            use_multi_scale=use_ms,
            use_rotation=use_rot,
            conf_thresh=conf_thresh
        )

        print(f"Counting Complete!")
        self.draw_results(results)
        self.count_var.set(f"TOTAL COUNT: {len(results)}")

    def draw_results(self, boxes, is_refresh=False):
        if not is_refresh:
            self.clear_results()
            self.current_results = boxes

        ratio = self.processor.ui_scale_ratio

        for box in boxes:
            x1, y1, x2, y2, score = box
            ui_x1 = int(x1 * ratio) + self.img_draw_x
            ui_y1 = int(y1 * ratio) + self.img_draw_y
            ui_x2 = int(x2 * ratio) + self.img_draw_x
            ui_y2 = int(y2 * ratio) + self.img_draw_y

            self.main_canvas.create_rectangle(ui_x1, ui_y1, ui_x2, ui_y2, outline="#00FF00", width=2, tags="result_box")
            center_x, center_y = (ui_x1 + ui_x2) // 2, (ui_y1 + ui_y2) // 2
            self.main_canvas.create_oval(center_x - 2, center_y - 2, center_x + 2, center_y + 2, fill="#00FF00", outline="#00FF00", tags="result_box")

        if not is_refresh:
            messagebox.showinfo("Success", f"Counting Complete!\n\nFound {len(boxes)} objects matching your template.")

# [HistoryTab class remains exactly as you had it]
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
        
        headings = ["ID", "Time", "Image Name", "Mode", "Conf.", "NMS Overlap", "Total Count"]
        for col, heading in zip(columns, headings):
            self.tree.heading(col, text=heading)
            self.tree.column(col, anchor=tk.CENTER, width=100)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)