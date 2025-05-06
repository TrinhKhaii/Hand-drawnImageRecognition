"""
@author: Trinh Khai Truong 
"""
import argparse
from collections import deque
import cv2
import numpy as np
import torch
import time
from camera_app.WebcamStream import WebcamStream
from src.constants import *
from src.model import QuickDraw
from src.utils import *


def get_args():
    parser = argparse.ArgumentParser("Hand-drawn Image Recognition")
    parser.add_argument("--pointer-color", type=str, choices=["green", "blue"], default="green", help="Color of the object used as the drawing pointer")
    parser.add_argument("--min-detect-area", type=int, default=3000, help="Minimum area (in pixels) of the detected pointer object to be considered valid")
    parser.add_argument("--canvas", type=bool, default=False, help="Display canvas")
    args = parser.parse_args()
    return args

    
def draw_points_on_canvas_and_frame(canvas, frame, points):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)
        cv2.line(frame, points[i - 1], points[i], GREEN_RGB, 2)


def get_center_of_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
    return center


def main(opt):

    if opt.pointer_color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB    
    

    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    is_drawing = False
    is_shown = False
    predicted_class = None
    too_small_warning = False
    warning_time = 0

    cam_stream = WebcamStream(0).start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuickDraw(num_classes=len(CLASSES))
    model.load_state_dict(torch.load('trained_models2/best_checkpoint.pt', weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    last_predict_time = 0
    predict_interval = 1.0

    if opt.canvas:
        cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canvas", 640, 480)

    while True:
        frame = cam_stream.read()
        if frame is None:
            continue

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            print("Quit!")
            break
        elif key == 32:
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points.clear()
                    canvas.fill(0)
                is_shown = False

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours_mask, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        if contours_mask:
            contour_pen = max(contours_mask, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(contour_pen)
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            if is_drawing:
                center = get_center_of_contour(contour_pen)
                if center:
                    points.appendleft(center)

        draw_points_on_canvas_and_frame(canvas, frame, points)
        

        current_time = time.time()
        if not is_drawing and not is_shown and len(points) > 0 and (current_time - last_predict_time) > predict_interval:
            last_predict_time = current_time
            image = preprocess_image(canvas, opt.min_detect_area)
            if image is not None:
                image_tensor = image.to(device)
                with torch.no_grad():
                    pred = model(image_tensor)
                    pred_idx = torch.argmax(pred).item()
                    predicted_class = CLASSES[pred_idx]
                    print(f"Prediction: {predicted_class}")
                    is_shown = True
                too_small_warning = False
            else:
                points.clear()
                canvas.fill(0)
                too_small_warning = True
                warning_time = current_time

        if is_shown and predicted_class is not None:
            cv2.putText(frame, f"You are drawing: {predicted_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_pointer, 2, cv2.LINE_AA)

        if too_small_warning and (current_time - warning_time < 2):
            cv2.putText(frame, "The object drawn is too small!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            too_small_warning = False

        cv2.imshow("Camera", frame)

        if opt.canvas:
            cv2.imshow("Canvas", 255 - canvas)

    cam_stream.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
