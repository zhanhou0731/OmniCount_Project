from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from core.image_processor import ImageProcessor
from core.matcher import MatcherEngine
import cv2
from core.database import DatabaseManager
import os
from utils.pdf_generator import PDFReportGenerator

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

        self.db = DatabaseManager()
        self.saved_orig_path = None
        self.saved_temp_path = None

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
            self.saved_orig_path = self.db.save_original_image(file_path)
            self.saved_temp_path = None
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
            self.saved_temp_path = self.db.save_template_image(template)
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

        db = DatabaseManager()
        
        params = []
        if filter_type != "none": params.append(f"Filter:{filter_type}")
        if use_clahe: params.append("CLAHE")
        if use_ms: params.append("Multi-Scale")
        if use_rot: params.append("Rotation")
        params_str = ", ".join(params) if params else "None"

        self.db.save_record(
            saved_orig_path=self.saved_orig_path,
            saved_temp_path=self.saved_temp_path,
            algorithm=algo,
            mode=img_mode,
            params_str=params_str,
            conf=conf_thresh,
            nms=nms_thresh,
            total_count=len(results),
            boxes=results
        )

        history_tab = self.master.children.get('!historytab')
        if history_tab:
            history_tab.refresh_table()

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


class HistoryTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.db = DatabaseManager()
        self.setup_ui()
        self.refresh_table()

    def setup_ui(self):
        action_frame = ttk.Frame(self, padding=10)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        left_action_frame = ttk.Frame(action_frame)
        left_action_frame.pack(side=tk.LEFT)

        ttk.Button(left_action_frame, text="Export Selected to PDF", command=self.export_pdf).pack(side=tk.LEFT, padx=5)

        ttk.Label(left_action_frame, text="💡 Hint: Hold Ctrl or Shift to select multiple rows", 
                  font=("Arial", 9, "italic"), foreground="gray").pack(side=tk.LEFT, padx=10)
        
        ttk.Button(action_frame, text="Clear History", command=self.clear_history).pack(side=tk.RIGHT, padx=5)

        columns = ("id", "time", "image", "algo", "mode", "conf", "nms", "count")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        
        headings = ["ID", "Time", "Image Name", "Algorithm", "Mode", "Conf.", "NMS", "Total Count"]
        widths = [40, 140, 120, 200, 80, 50, 50, 80]
        for col, heading, w in zip(columns, headings, widths):
            self.tree.heading(col, text=heading)
            self.tree.column(col, anchor=tk.CENTER, width=w)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<Double-1>", self.on_row_double_click)


    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        records = self.db.get_all_records()
        
        for record in records:
            r_id, r_time, r_img_path, r_algo, r_mode, r_conf, r_nms, r_count = record
            
            filename = os.path.basename(r_img_path)
            
            self.tree.insert("", tk.END, values=(r_id, r_time, filename, r_algo, r_mode, f"{r_conf:.2f}", f"{r_nms:.2f}", r_count))

    def on_row_double_click(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            return
            
        row_values = self.tree.item(selected_item[0])['values']
        record_id = row_values[0] 
        
        full_record = self.db.get_record_by_id(record_id)
        
        if full_record:
            self.show_preview_window(full_record)
        else:
            messagebox.showerror("Error", "Could not load record details from database.")


    def show_preview_window(self, record):
        
        r_id, r_time, r_orig, r_temp, r_algo, r_mode, r_params, r_conf, r_nms, r_count, r_boxes_json = record
        
        preview_win = tk.Toplevel(self)
        preview_win.title(f"Detailed Report: Record #{r_id}")
        preview_win.geometry("900x600")
        preview_win.configure(bg="#2b2b2b")

        # Layout: Left side for image, Right side for text
        left_frame = ttk.Frame(preview_win, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # ---  Scrollable Right Panel ---
        right_container = ttk.Frame(preview_win)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)

        right_canvas = tk.Canvas(right_container, highlightthickness=0, width=280)
        right_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_frame = ttk.Frame(right_canvas, padding=20)
        right_frame_id = right_canvas.create_window((0, 0), window=right_frame, anchor="nw")

        right_frame.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>", lambda e: right_canvas.itemconfig(right_frame_id, width=e.width))

        # Enable Mousewheel scrolling
        def _on_mousewheel(event):
            if right_frame.winfo_reqheight() > right_canvas.winfo_height():
                right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        preview_win.bind("<MouseWheel>", _on_mousewheel)
        
        try:
            temp_processor = ImageProcessor(max_width=1200)
            temp_processor.load_image(r_orig)
            img_bgr = temp_processor.original_image.copy()
            
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image at: {r_orig}")
            
            import json
            boxes = json.loads(r_boxes_json)
            
            for box in boxes:
                x1, y1, x2, y2, score = box
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            h, w = img_bgr.shape[:2]
            scale = min(600/w, 600/h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))
            
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            img_lbl = tk.Label(left_frame, image=tk_img, bg="#2b2b2b")
            img_lbl.image = tk_img # Prevent garbage collection!
            img_lbl.pack(expand=True)
            
        except Exception as e:
            tk.Label(left_frame, text=f"Image Error:\n{str(e)}", fg="red", bg="#2b2b2b").pack(expand=True)

        #right panel
        title_font = ("Arial", 14, "bold")
        data_font = ("Arial", 11)
        
        ttk.Label(right_frame, text="Execution Summary", font=title_font).pack(anchor=tk.W, pady=(0, 10))
        
        # ---  Display the Template Image ---
        ttk.Label(right_frame, text="Template Searched:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        try:
            temp_bgr = cv2.imread(r_temp)
            if temp_bgr is not None:
                temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
                pil_temp = Image.fromarray(temp_rgb)
                pil_temp.thumbnail((100, 100)) 
                tk_temp = ImageTk.PhotoImage(pil_temp)
                
                temp_lbl = tk.Label(right_frame, image=tk_temp, bg="#2b2b2b", highlightthickness=1, highlightbackground="#555555")
                temp_lbl.image = tk_temp # Prevent garbage collection!
                temp_lbl.pack(anchor=tk.W, pady=(5, 15))
            else:
                ttk.Label(right_frame, text="[Template file missing]").pack(anchor=tk.W, pady=(5, 15))
        except Exception as e:
            print(f"Preview Template Error: {e}")

        # The Parameter List
        details = [
            ("Time:", r_time),
            ("Algorithm:", r_algo),
            ("Image Mode:", r_mode.title()),
            ("Parameters:", r_params),
            ("Confidence Threshold:", f"{r_conf:.2f}"),
            ("NMS Overlap:", f"{r_nms:.2f}"),
        ]
        
        for label, val in details:
            ttk.Label(right_frame, text=label, font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(5,0))
            ttk.Label(right_frame, text=str(val), font=data_font, wraplength=200).pack(anchor=tk.W)
            
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        ttk.Label(right_frame, text="FINAL COUNT", font=("Arial", 12, "bold")).pack(anchor=tk.CENTER)
        count_lbl = tk.Label(right_frame, text=str(r_count), font=("Arial", 48, "bold"), fg="#00FF00")
        count_lbl.pack(anchor=tk.CENTER, pady=10)


    def export_pdf(self):
        selected_items = self.tree.selection()
        
        if not selected_items:
            messagebox.showwarning("Warning", "Please select at least one record to export!\n(Hold Ctrl/Shift to select multiple)")
            return
            
        record_ids = []
        for item in selected_items:
            row_values = self.tree.item(item)['values']
            record_ids.append(row_values[0]) # ID is the first column
            
        default_name = f"OmniCount_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        filepath = filedialog.asksaveasfilename(
            title="Save PDF Report",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if not filepath:
            return
            
        try:
            pdf_gen = PDFReportGenerator(self.db)
            pdf_gen.generate_report(record_ids, filepath)
            
            messagebox.showinfo("Success", f"PDF Report successfully generated!\nSaved to: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate PDF:\n{str(e)}")


    def clear_history(self):

        confirm = messagebox.askyesno(
            "Clear History", 
            "Are you sure you want to delete ALL history records and saved images?\n\nThis cannot be undone."
        )
        
        if confirm:
            try:
                self.db.clear_all_records()
                
                self.refresh_table()
                
                messagebox.showinfo("Success", "History and saved files have been completely cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear history:\n{str(e)}")