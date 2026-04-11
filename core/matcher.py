import cv2
import numpy as np
import imutils
import concurrent.futures

class MatcherEngine:
    def __init__(self):
        self.matched_boxes = []

    def match(self, image, template, algorithm="Normalized Cross-Correlation (NCC)",
              image_mode="grayscale", filter_type="none", use_clahe=False, nms_thresh=0.3,
              use_multi_scale=True, use_rotation=False, conf_thresh=0.8):

        # 1. PRE-PROCESSING
        work_img, work_temp = self._preprocess(image, template, image_mode, filter_type, use_clahe)

        # 2. RUN DENSE MATCHER 
        raw_boxes = self._run_dense_matcher(work_img, work_temp, algorithm, conf_thresh, use_multi_scale, use_rotation)

        # 3. NON-MAXIMUM SUPPRESSION (NMS)
        self.matched_boxes = self._non_max_suppression(raw_boxes, nms_thresh)
        return self.matched_boxes

    def _preprocess(self, img, temp, image_mode, filter_type, use_clahe):
        w_img = img.copy()
        w_temp = temp.copy()

        #  Noise Filters
        w_img = self._apply_filter(w_img, filter_type)
        w_temp = self._apply_filter(w_temp, filter_type)

        #  CLAHE
        if use_clahe:
            w_img = self._apply_clahe(w_img)
            w_temp = self._apply_clahe(w_temp)

        # C. Image Mode (Grayscale, Color, or Edges)
        if image_mode in ["grayscale", "edges"]:
            if len(w_img.shape) == 3: w_img = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
            if len(w_temp.shape) == 3: w_temp = cv2.cvtColor(w_temp, cv2.COLOR_BGR2GRAY)

            if image_mode == "edges":
                w_img = cv2.Canny(w_img, 50, 150)
                w_temp = cv2.Canny(w_temp, 50, 150)

        return w_img, w_temp

    def _apply_filter(self, img, filter_type):
        if filter_type == "gaussian":
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif filter_type == "bilateral":
            return cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def _apply_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 2: # Grayscale
            return clahe.apply(img)
        else: 
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _run_dense_matcher(self, work_img, work_temp, algorithm, conf_thresh, use_multi_scale, use_rotation):
        angles_to_check = range(0, 360, 15) if use_rotation else [0]
        raw_boxes = []

        def check_angle(angle):
            rotated_temp = work_temp if angle == 0 else imutils.rotate_bound(work_temp, angle)
            if use_multi_scale:
                return self._multi_scale_match(work_img, rotated_temp, conf_thresh, algorithm)
            else:
                return self._single_scale_match(work_img, rotated_temp, conf_thresh, algorithm)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread_results = executor.map(check_angle, angles_to_check)
            for boxes in thread_results:
                raw_boxes.extend(boxes)

        return raw_boxes

    
    def _extract_top_matches(self, res, w, h, conf_thresh, algorithm, limit=1000):
        boxes = []
        is_ncc = (algorithm == "Normalized Cross-Correlation (NCC)")

        if is_ncc:
            valid_y, valid_x = np.where(res >= conf_thresh)
        else: 
            valid_y, valid_x = np.where(res <= (1.0 - conf_thresh))

        if len(valid_y) > limit:
            scores = res[valid_y, valid_x]
            if is_ncc:
                best_idx = np.argsort(scores)[-limit:]
            else:
                best_idx = np.argsort(scores)[:limit]
            valid_y, valid_x = valid_y[best_idx], valid_x[best_idx]

        for x, y in zip(valid_x, valid_y):
            score = res[y, x] if is_ncc else (1.0 - res[y, x])
            boxes.append([x, y, x + w, y + h, score])
            
        return boxes

    def _single_scale_match(self, img, template, conf_thresh, algorithm):
        h, w = template.shape[:2]
        if w > img.shape[1] or h > img.shape[0]: return []
        
        meth = cv2.TM_CCOEFF_NORMED if algorithm == "Normalized Cross-Correlation (NCC)" else cv2.TM_SQDIFF_NORMED
        res = cv2.matchTemplate(img, template, meth)
        
        return self._extract_top_matches(res, w, h, conf_thresh, algorithm)

    def _multi_scale_match(self, img, template, conf_thresh, algorithm):
        boxes = []
        scales = np.linspace(0.5, 1.5, 10)
        meth = cv2.TM_CCOEFF_NORMED if algorithm == "Normalized Cross-Correlation (NCC)" else cv2.TM_SQDIFF_NORMED
        
        for scale in scales:
            t_w = int(template.shape[1] * scale)
            t_h = int(template.shape[0] * scale)
            if t_w > img.shape[1] or t_h > img.shape[0] or t_w == 0 or t_h == 0: continue
            
            resized_template = cv2.resize(template, (t_w, t_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(img, resized_template, meth)
            
            boxes.extend(self._extract_top_matches(res, t_w, t_h, conf_thresh, algorithm))
            
        return boxes

    def _non_max_suppression(self, boxes, nms_thresh):
        if len(boxes) == 0: return []
        boxes = np.array(boxes)
        pick = []
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)

        while len(order) > 0:
            last = len(order) - 1
            i = order[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[order[:last]])
            yy1 = np.maximum(y1[i], y1[order[:last]])
            xx2 = np.minimum(x2[i], x2[order[:last]])
            yy2 = np.minimum(y2[i], y2[order[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / (areas[i] + areas[order[:last]] - (w * h))
            order = np.delete(order, np.concatenate(([last], np.where(overlap > nms_thresh)[0])))

        return boxes[pick].astype("int").tolist()