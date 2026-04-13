import os
import json
import cv2
from datetime import datetime
from fpdf import FPDF
from core.image_processor import ImageProcessor

class PDFReportGenerator:    
    def __init__(self, db_manager):
        self.db = db_manager

    def generate_report(self, record_ids, output_filepath):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        records = [self.db.get_record_by_id(r_id) for r_id in record_ids]
        
        self._build_summary_page(pdf, records)

        for record in records:
            self._build_detail_page(pdf, record)

        pdf.output(output_filepath)
        print(f"[PDF] Report successfully saved to {output_filepath}")

    def _build_summary_page(self, pdf, records):
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 15, "OmniCount Execution Report", ln=True, align='C')
        
        pdf.set_font("Arial", 'I', 12)
        date_str = datetime.now().strftime("%B %d, %Y - %H:%M")
        pdf.cell(0, 10, f"Generated on: {date_str}", ln=True, align='C')
        pdf.ln(10)

        # Table Header
        pdf.set_font("Arial", 'B', 11)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(20, 10, "ID", border=1, fill=True, align='C')
        pdf.cell(50, 10, "Time", border=1, fill=True, align='C')
        pdf.cell(85, 10, "Algorithm", border=1, fill=True, align='C')
        pdf.cell(35, 10, "Total Count", border=1, fill=True, align='C')
        pdf.ln()

        # Table Rows
        pdf.set_font("Arial", '', 10)
        for r in records:
            r_id, r_time, r_orig, r_temp, r_algo, r_mode, r_params, r_conf, r_nms, r_count, r_json = r
            pdf.cell(20, 10, str(r_id), border=1, align='C')
            pdf.cell(50, 10, r_time, border=1, align='C')
            pdf.cell(85, 10, r_algo[:35], border=1, align='C')
            pdf.cell(35, 10, str(r_count), border=1, align='C')
            pdf.ln()

    def _build_detail_page(self, pdf, record):
        pdf.add_page()
        r_id, r_time, r_orig, r_temp, r_algo, r_mode, r_params, r_conf, r_nms, r_count, r_boxes_json = record

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Detailed Analysis: Record #{r_id}", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 5, f"Execution Time: {r_time}", ln=True)
        pdf.ln(5)

        temp_result_path = f"temp_pdf_render_{r_id}.jpg"
        try:
            temp_processor = ImageProcessor(max_width=1200)
            temp_processor.load_image(r_orig)
            img_bgr = temp_processor.original_image.copy()

            if img_bgr is None:
                raise FileNotFoundError("Could not load image.")

            abs_boxes = json.loads(r_boxes_json)
            rel_boxes = temp_processor.from_original_coords(abs_boxes)
            
            # Draw the boxes
            for box in rel_boxes:
                x1, y1, x2, y2, score = box
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
            cv2.imwrite(temp_result_path, img_bgr)

            # Insert Image into PDF 
            pdf.image(temp_result_path, w=150)
            
        except Exception as e:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Image Render Error: {str(e)}", ln=True)
            pdf.set_text_color(0, 0, 0)

        pdf.ln(5)

        # --- PARAMETERS & TEMPLATE SECTION ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Configuration & Results:", ln=True)
        
        pdf.set_font("Arial", '', 11)
        pdf.cell(60, 6, "Algorithm Used:", border=0)
        pdf.cell(0, 6, r_algo, ln=True)
        
        pdf.cell(60, 6, "Image Mode:", border=0)
        pdf.cell(0, 6, r_mode.title(), ln=True)
        
        pdf.cell(60, 6, "Enhancements:", border=0)
        pdf.cell(0, 6, r_params, ln=True)
        
        pdf.cell(60, 6, "Confidence Threshold:", border=0)
        pdf.cell(0, 6, f"{r_conf:.2f}", ln=True)
        
        pdf.cell(60, 6, "NMS Threshold:", border=0)
        pdf.cell(0, 6, f"{r_nms:.2f}", ln=True)
        pdf.ln(5)

        #  COUNT
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(65, 8, "TOTAL OBJECTS FOUND:")
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 150, 0) # Dark green
        pdf.cell(0, 8, str(r_count), ln=True)
        pdf.set_text_color(0, 0, 0)

        #  template image
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 5, "Template Searched:", ln=True)
        if os.path.exists(r_temp):
            pdf.image(r_temp, x=15, w=30) 
        
        # --- CLEANUP ---
        if os.path.exists(temp_result_path):
            os.remove(temp_result_path)