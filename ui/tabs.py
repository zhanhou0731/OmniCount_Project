import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from datetime import datetime
from typing import Optional, List, Any

from core.image_processor import ImageProcessor
from core.matcher import MatcherEngine
from core.database import DatabaseManager
from ui.image_canvas import ImageCanvas
from utils.pdf_generator import PDFReportGenerator

class MainCounterTab(ttk.Frame):
    def __init__(self, parent, count_var: tk.StringVar):
        super().__init__(parent)
        self.count_var = count_var
        self.processor = ImageProcessor(max_width=1200)
        self.engine = MatcherEngine()
        self.db = DatabaseManager()
        
        self.current_image_path: Optional[str] = None
        self.saved_orig_path: Optional[str] = None
        self.saved_temp_path: Optional[str] = None
        self.resize_timer: Optional[str] = None

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

    def setup_ui(self):
        # --- Left Panel (Control Panel) ---
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        self.control_canvas = tk.Canvas(left_container, highlightthickness=0, width=260)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.control_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        self.control_frame = ttk.Frame(self.control_canvas, padding=10)
        self.control_frame_id = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        self.control_frame.bind("<Configure>", self._on_frame_configure)
        self.control_canvas.bind("<Configure>", self._on_canvas_configure)
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._setup_control_widgets()

        # --- Right Panel (Image Canvas) ---
        canvas_frame = ttk.Frame(self, padding=10)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_canvas = ImageCanvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.create_text(300, 250, text="Upload an image to begin...", fill="white", font=("Arial", 14))
        
        self.image_canvas.bind("<Configure>", self._on_canvas_resize)

    def _setup_control_widgets(self):
        cf = self.control_frame
        
        # Setup Section
        ttk.Label(cf, text="Setup", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(cf, text="1. Upload Image", command=self.upload_image).pack(fill=tk.X, pady=2)
        self.btn_select_template = ttk.Button(cf, text="2. Select Template", command=self.toggle_crop_mode)
        self.btn_select_template.pack(fill=tk.X, pady=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Parameters Section
        ttk.Label(cf, text="Parameters", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(cf, text="Algorithm:").pack(anchor=tk.W)
        self.algo_dropdown = ttk.Combobox(cf, textvariable=self.algo_var, state="readonly")
        self.algo_dropdown['values'] = (self.ncc_name, self.ssd_name)
        self.algo_dropdown.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(cf, text="Image Mode:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Radiobutton(cf, text="Grayscale (Fast)", variable=self.img_mode_var, value="grayscale").pack(anchor=tk.W)
        ttk.Radiobutton(cf, text="Color (RGB)", variable=self.img_mode_var, value="color").pack(anchor=tk.W)
        ttk.Radiobutton(cf, text="Edges", variable=self.img_mode_var, value="edges").pack(anchor=tk.W)

        ttk.Label(cf, text="Noise Filter:").pack(anchor=tk.W, pady=(5, 0))
        filter_dropdown = ttk.Combobox(cf, textvariable=self.filter_var, state="readonly")
        filter_dropdown['values'] = ("none", "gaussian", "bilateral")
        filter_dropdown.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(cf, text="Enhancements:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Checkbutton(cf, text="CLAHE", variable=self.clahe_var).pack(anchor=tk.W, pady=(5, 5))
        ttk.Checkbutton(cf, text="Multi-Scale Matching", variable=self.multi_scale_var).pack(anchor=tk.W)
        ttk.Checkbutton(cf, text="Rotation Invariance", variable=self.rotation_var).pack(anchor=tk.W, pady=(0, 5))

        # Sliders
        self.conf_lbl_var = tk.StringVar(value="Confidence Threshold: 0.80")
        ttk.Label(cf, textvariable=self.conf_lbl_var).pack(anchor=tk.W, pady=(5, 0))
        self.conf_slider = ttk.Scale(cf, from_=0.1, to=1.0, value=0.8, orient=tk.HORIZONTAL,
                                    command=lambda v: self.conf_lbl_var.set(f"Confidence Threshold: {float(v):.2f}"))
        self.conf_slider.pack(fill=tk.X)

        self.nms_lbl_var = tk.StringVar(value="NMS Overlap Threshold: 0.30")
        ttk.Label(cf, textvariable=self.nms_lbl_var).pack(anchor=tk.W, pady=(10, 0))
        self.nms_slider = ttk.Scale(cf, from_=0.1, to=1.0, value=0.3, orient=tk.HORIZONTAL,
                                    command=lambda v: self.nms_lbl_var.set(f"NMS Overlap Threshold: {float(v):.2f}"))
        self.nms_slider.pack(fill=tk.X)

        # Previews & Execution
        ttk.Label(cf, text="Template Preview:", font=("Arial", 9)).pack(anchor=tk.W, pady=(10, 0))
        self.template_canvas = tk.Canvas(cf, width=100, height=100, bg="#333333", highlightthickness=1)
        self.template_canvas.pack(pady=5)
        
        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(cf, text="Execute", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(cf, text="START COUNTING", command=self.execute_counting).pack(fill=tk.X, pady=5)
        ttk.Button(cf, text="Clear Results", command=self.clear_results).pack(fill=tk.X)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image to Count",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not file_path: return
        try:
            self.processor.load_image(file_path)
            self.current_image_path = file_path
            self.saved_orig_path = self.db.save_original_image(file_path)
            self.saved_temp_path = None
            self.image_canvas.crop_coords = None
            self.image_canvas.set_image(self.processor.original_image)
            self.clear_results()
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to load image:\n{str(e)}")

    def toggle_crop_mode(self):
        if self.processor.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
            
        if not self.image_canvas.crop_mode_active:
            self.clear_results()
            self.image_canvas.enable_crop_mode(self._on_template_selected)
            self.btn_select_template.config(text="🔴 Cancel Selection")
        else:
            self.image_canvas.disable_crop_mode()
            self.btn_select_template.config(text="2. Select Template")

    def _on_template_selected(self, x1, y1, x2, y2):
        self.btn_select_template.config(text="2. Select Template")
        template = self.processor.crop_template(x1, y1, x2, y2)
        if template is not None:
            self.saved_temp_path = self.db.save_template_image(template)
            self._update_template_preview()

    def _update_template_preview(self):
        temp_bgr = self.processor.template_image
        if temp_bgr is None: return
        temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(temp_rgb)
        pil_img.thumbnail((100, 100)) 
        self.tk_template_image = ImageTk.PhotoImage(pil_img) 
        self.template_canvas.delete("all")
        self.template_canvas.create_image(50, 50, anchor=tk.CENTER, image=self.tk_template_image)

    def clear_results(self):
        self.image_canvas.clear_results()
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

        # 1. Preprocess
        work_img, work_temp = self.processor.get_processed_main_and_template(
            mode=img_mode, filter_type=filter_type, use_clahe=use_clahe
        )

        # 2. Match
        results = self.engine.match(
            work_img=work_img,
            work_temp=work_temp,
            algorithm=algo,
            nms_thresh=nms_thresh,
            use_multi_scale=use_ms,
            use_rotation=use_rot,
            conf_thresh=conf_thresh
        )

        # 3. Update UI
        self.image_canvas.draw_results(results)
        self.count_var.set(f"TOTAL COUNT: {len(results)}")
        messagebox.showinfo("Success", f"Counting Complete!\n\nFound {len(results)} objects.")

        # 4. Save to Database
        params = []
        if filter_type != "none": params.append(f"Filter:{filter_type}")
        if use_clahe: params.append("CLAHE")
        if use_ms: params.append("Multi-Scale")
        if use_rot: params.append("Rotation")
        params_str = ", ".join(params) if params else "None"

        # Map results back to original image size for persistent storage
        abs_results = self.processor.to_original_coords(results)

        self.db.save_record(
            saved_orig_path=self.saved_orig_path,
            saved_temp_path=self.saved_temp_path,
            algorithm=algo,
            mode=img_mode,
            params_str=params_str,
            conf=conf_thresh,
            nms=nms_thresh,
            total_count=len(results),
            boxes=abs_results
        )

        # Notify History Tab
        history_tab = self.master.children.get('!historytab')
        if history_tab:
            history_tab.refresh_table()

    def _on_canvas_resize(self, event):
        if self.processor.original_image is not None:
            if self.resize_timer is not None: self.after_cancel(self.resize_timer)
            self.resize_timer = self.after(150, self.image_canvas.refresh)

    def _on_frame_configure(self, event):
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.control_canvas.itemconfig(self.control_frame_id, width=event.width)

    def _on_mousewheel(self, event):
        frame_height = self.control_frame.winfo_reqheight()
        canvas_height = self.control_canvas.winfo_height()
        if frame_height > canvas_height:
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class HistoryTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.db = DatabaseManager()
        self.setup_ui()
        self.refresh_table()

    def setup_ui(self):
        action_frame = ttk.Frame(self, padding=10)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(action_frame, text="Export Selected to PDF", command=self.export_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Label(action_frame, text="💡 Hint: Hold Ctrl or Shift to select multiple rows", 
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
        if not selected_item: return
        
        record_id = self.tree.item(selected_item[0])['values'][0]
        record = self.db.get_record_by_id(record_id)
        if record:
            self.show_preview_window(record)

    def show_preview_window(self, record):
        r_id, r_time, r_orig, r_temp, r_algo, r_mode, r_params, r_conf, r_nms, r_count, r_boxes_json = record
        
        preview_win = tk.Toplevel(self)
        preview_win.title(f"Detailed Report: Record #{r_id}")
        preview_win.geometry("1000x650")
        preview_win.configure(bg="#2b2b2b")

        left_frame = ttk.Frame(preview_win, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Use ImageCanvas for preview!
        preview_canvas = ImageCanvas(left_frame, bg="#2b2b2b", highlightthickness=0)
        preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Ensure it refreshes when window is ready
        preview_canvas.bind("<Configure>", lambda e: preview_canvas.refresh())
        
        right_container = ttk.Frame(preview_win, width=300)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)

        # Scrollable info panel
        right_canvas = tk.Canvas(right_container, highlightthickness=0, width=280)
        right_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_frame = ttk.Frame(right_canvas, padding=20)
        right_frame_id = right_canvas.create_window((0, 0), window=right_frame, anchor="nw")
        right_frame.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>", lambda e: right_canvas.itemconfig(right_frame_id, width=e.width))

        # Load and display results
        try:
            temp_proc = ImageProcessor(max_width=1200)
            img = temp_proc.load_image(r_orig)
            import json
            abs_boxes = json.loads(r_boxes_json)
            
            # Map back to safety-resized scale
            rel_boxes = temp_proc.from_original_coords(abs_boxes)
            
            preview_canvas.set_image(img)
            preview_canvas.draw_results(rel_boxes)
            
            # Re-trigger refresh after a short delay to ensure dimensions are correct
            preview_win.after(200, preview_canvas.refresh)
        except Exception as e:
            tk.Label(left_frame, text=f"Error loading image: {e}", fg="red").pack()

        # Summary text
        ttk.Label(right_frame, text="Execution Summary", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # Template Preview
        ttk.Label(right_frame, text="Template Searched:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        try:
            t_img = cv2.imread(r_temp)
            if t_img is not None:
                t_rgb = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
                pil_t = Image.fromarray(t_rgb)
                pil_t.thumbnail((100, 100))
                tk_t = ImageTk.PhotoImage(pil_t)
                lbl = tk.Label(right_frame, image=tk_t, bg="#2b2b2b")
                lbl.image = tk_t
                lbl.pack(anchor=tk.W, pady=5)
        except: pass

        details = [
            ("Time:", r_time),
            ("Algorithm:", r_algo),
            ("Image Mode:", r_mode.title()),
            ("Parameters:", r_params),
            ("Conf. Threshold:", f"{r_conf:.2f}"),
            ("NMS Overlap:", f"{r_nms:.2f}"),
        ]
        for label, val in details:
            ttk.Label(right_frame, text=label, font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(5,0))
            ttk.Label(right_frame, text=str(val), font=("Arial", 11), wraplength=200).pack(anchor=tk.W)
            
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        ttk.Label(right_frame, text="FINAL COUNT", font=("Arial", 12, "bold")).pack(anchor=tk.CENTER)
        tk.Label(right_frame, text=str(r_count), font=("Arial", 48, "bold"), fg="#00FF00").pack(anchor=tk.CENTER)

    def export_pdf(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Select at least one record!")
            return
            
        record_ids = [self.tree.item(i)['values'][0] for i in selected_items]
        
        default_name = f"OmniCount_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        filepath = filedialog.asksaveasfilename(
            title="Save PDF Report",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if filepath:
            try:
                PDFReportGenerator(self.db).generate_report(record_ids, filepath)
                messagebox.showinfo("Success", f"Report saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate PDF: {e}")

    def clear_history(self):
        if messagebox.askyesno("Clear History", "Delete ALL records?"):
            self.db.clear_all_records()
            self.refresh_table()
