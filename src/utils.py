"""
@author: Trinh Khai Truong 
"""
import cv2
import torch
import numpy as np


def preprocess_image(canvas, min_area):
    canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(canvas_gs, 5)

    gaussian = cv2.GaussianBlur(median, (3, 3), 0)
    _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = canvas_gs[y:y + h, x:x + w]
    roi_resized = cv2.resize(roi, (28, 28))

    img_array = roi_resized.astype(np.float32) / 255.0
    img_array = img_array[None, None, :, :] 
    tensor = torch.from_numpy(img_array)
    return tensor
