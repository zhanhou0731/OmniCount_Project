import sqlite3
import os
import shutil
import json
import cv2
from datetime import datetime
import uuid

class DatabaseManager:
    
    def __init__(self, base_dir="history"):
        self.base_dir = base_dir
        self.db_path = os.path.join(self.base_dir, "omnicount.db")
        self.img_dir = os.path.join(self.base_dir, "original_images")
        self.temp_dir = os.path.join(self.base_dir, "templates")
        self.report_dir = os.path.join(self.base_dir, "exported_reports")

        self._setup_environment()

    def _setup_environment(self):
        for directory in [self.base_dir, self.img_dir, self.temp_dir, self.report_dir]:
            os.makedirs(directory, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                template_path TEXT,
                algorithm TEXT,
                mode TEXT,
                parameters TEXT,
                conf_thresh REAL,
                nms_thresh REAL,
                total_count INTEGER,
                box_data TEXT
            )
        ''')
        conn.commit()
        conn.close()

    
    def save_original_image(self, orig_img_path):
        file_suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        orig_ext = os.path.splitext(orig_img_path)[1]
        new_orig_path = os.path.join(self.img_dir, f"img_{file_suffix}{orig_ext}")
        shutil.copy(orig_img_path, new_orig_path)
        return new_orig_path

    def save_template_image(self, template_bgr_array):
        file_suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        new_temp_path = os.path.join(self.temp_dir, f"temp_{file_suffix}.png")
        cv2.imwrite(new_temp_path, template_bgr_array)
        return new_temp_path

    def save_record(self, saved_orig_path, saved_temp_path, algorithm, mode, params_str, conf, nms, total_count, boxes):
        display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        boxes_json = json.dumps(boxes)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history_records 
            (timestamp, image_path, template_path, algorithm, mode, parameters, conf_thresh, nms_thresh, total_count, box_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (display_time, saved_orig_path, saved_temp_path, algorithm, mode, params_str, conf, nms, total_count, boxes_json))
        
        conn.commit()
        conn.close()
        print(f"[DATABASE] Record saved successfully. ID: {cursor.lastrowid}")

    def get_all_records(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, image_path, algorithm, mode, conf_thresh, nms_thresh, total_count FROM history_records ORDER BY id DESC")
        records = cursor.fetchall()
        conn.close()
        return records
    
    def get_record_by_id(self, record_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_records WHERE id=?", (record_id,))
        record = cursor.fetchone()
        conn.close()
        return record
    

    def clear_all_records(self):
        for directory in [self.img_dir, self.temp_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"[DATABASE] Failed to delete file {file_path}: {e}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history_records")
        conn.commit()
        conn.close()
        print("[DATABASE] All history records and files have been cleared.")
    
    