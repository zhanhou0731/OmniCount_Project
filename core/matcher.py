import cv2
import numpy as np
import imutils
import concurrent.futures

class MatcherEngine:    
    def __init__(self):
        self.matched_boxes = [] 

    def match(self, image, template, conf_thresh=0.8, nms_thresh=0.3, 
              use_multi_scale=True, use_rotation=False, filter_type="none", is_grayscale=True):

        # Grayscale/color
        if is_grayscale:
            if len(image.shape) == 3:
                work_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                work_img = image.copy()
                
            if len(template.shape) == 3:
                work_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                work_temp = template.copy()
        else:
            work_img = image.copy()
            work_temp = template.copy()

        # Noise Filtering
        work_img = self._apply_filter(work_img, filter_type)
        work_temp = self._apply_filter(work_temp, filter_type)

        # rotation angles
        angles_to_check = range(0, 360, 15) if use_rotation else [0]
        
        raw_boxes = []
        
        def check_angle(angle):
            if angle == 0:
                rotated_temp = work_temp
            else:
                rotated_temp = imutils.rotate_bound(work_temp, angle)
                
            if use_multi_scale:
                return self._multi_scale_match(work_img, rotated_temp, conf_thresh)
            else:
                return self._single_scale_match(work_img, rotated_temp, conf_thresh)

        # Thread Pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread_results = executor.map(check_angle, angles_to_check)
            
            for boxes in thread_results:
                raw_boxes.extend(boxes)

        # NMS
        self.matched_boxes = self._non_max_suppression(raw_boxes, nms_thresh)
        
        return self.matched_boxes

    def _apply_filter(self, img, filter_type):
        if filter_type == "gaussian":
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif filter_type == "bilateral":
            return cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def _single_scale_match(self, img, template, conf_thresh):
        boxes = []
        h, w = template.shape[:2]
        
        if w > img.shape[1] or h > img.shape[0]:
            return boxes
            
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= conf_thresh)
        
        for pt in zip(*loc[::-1]):
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]])
            
        return boxes

    def _multi_scale_match(self, img, template, conf_thresh):
        boxes = []
        scales = np.linspace(0.5, 1.5, 20) 
        
        for scale in scales:
            t_w = int(template.shape[1] * scale)
            t_h = int(template.shape[0] * scale)
            
            if t_w > img.shape[1] or t_h > img.shape[0] or t_w == 0 or t_h == 0:
                continue
                
            resized_template = cv2.resize(template, (t_w, t_h), interpolation=cv2.INTER_AREA)
            
            res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= conf_thresh)
            
            for pt in zip(*loc[::-1]):
                boxes.append([pt[0], pt[1], pt[0] + t_w, pt[1] + t_h, res[pt[1], pt[0]]])
                
        return boxes

    def _non_max_suppression(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

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