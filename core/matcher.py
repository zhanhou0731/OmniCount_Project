import cv2
import numpy as np
import imutils
import concurrent.futures
from typing import List

class MatcherEngine:
    """Core mathematics and computer vision engine for object counting."""
    
    def __init__(self):
        self.matched_boxes: List[List[int]] = []

    def match(self, 
              work_img: np.ndarray, 
              work_temp: np.ndarray, 
              algorithm: str = "Normalized Cross-Correlation (NCC)",
              nms_thresh: float = 0.3,
              use_multi_scale: bool = True, 
              use_rotation: bool = False, 
              conf_thresh: float = 0.8) -> List[List[int]]:
        """
        Executes template matching on pre-processed images.
        Returns a list of boxes [x1, y1, x2, y2, score].
        """
        # 1. RUN DENSE MATCHER 
        raw_boxes = self._run_dense_matcher(work_img, work_temp, algorithm, conf_thresh, use_multi_scale, use_rotation)

        # 2. NON-MAXIMUM SUPPRESSION (NMS)
        self.matched_boxes = self._non_max_suppression(raw_boxes, nms_thresh)
        return self.matched_boxes

    def _run_dense_matcher(self, 
                          work_img: np.ndarray, 
                          work_temp: np.ndarray, 
                          algorithm: str, 
                          conf_thresh: float, 
                          use_multi_scale: bool, 
                          use_rotation: bool) -> List[List[float]]:
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

    def _extract_top_matches(self, 
                             res: np.ndarray, 
                             w: int, 
                             h: int, 
                             conf_thresh: float, 
                             algorithm: str, 
                             limit: int = 1000) -> List[List[float]]:
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
            score = float(res[y, x] if is_ncc else (1.0 - res[y, x]))
            boxes.append([int(x), int(y), int(x + w), int(y + h), score])
            
        return boxes

    def _single_scale_match(self, 
                           img: np.ndarray, 
                           template: np.ndarray, 
                           conf_thresh: float, 
                           algorithm: str) -> List[List[float]]:
        h, w = template.shape[:2]
        if w > img.shape[1] or h > img.shape[0]: return []
        
        meth = cv2.TM_CCOEFF_NORMED if algorithm == "Normalized Cross-Correlation (NCC)" else cv2.TM_SQDIFF_NORMED
        res = cv2.matchTemplate(img, template, meth)
        
        return self._extract_top_matches(res, w, h, conf_thresh, algorithm)

    def _multi_scale_match(self, 
                          img: np.ndarray, 
                          template: np.ndarray, 
                          conf_thresh: float, 
                          algorithm: str) -> List[List[float]]:
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

    def _non_max_suppression(self, 
                             boxes: List[List[float]], 
                             nms_thresh: float) -> List[List[int]]:
        if not boxes: return []
        
        boxes_np = np.array(boxes)
        x1, y1, x2, y2, scores = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3], boxes_np[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)

        pick = []
        while len(order) > 0:
            i = order[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / (areas[i] + areas[order[:-1]] - (w * h))
            order = np.delete(order, np.concatenate(([len(order)-1], np.where(overlap > nms_thresh)[0])))

        return boxes_np[pick].astype("int").tolist()
