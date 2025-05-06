"""
@author: Trinh Khai Truong 
"""
import cv2
from collections import deque
import mediapipe as mp
import numpy as np
import torch
from camera_app.WebcamStream import WebcamStream
from src.constants import *
from src.model import QuickDraw
from src.utils import *
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_args():
    parser = argparse.ArgumentParser(description="Hand-drawn Image Recognition")
    parser.add_argument("--min-detect-area", type=int, default=3000, help="Minimum area (in pixels) of the detected pointer object to be considered valid")
    parser.add_argument("--prediction-display-time", type=int, default=3, help="Duration (in seconds) to display the prediction result on the screen")
    return parser.parse_args()

def fingers_touching(landmarks, threshold=0.05):
    # Check that your thumb and index finger are touching
    x1, y1 = landmarks.landmark[4].x, landmarks.landmark[4].y
    x2, y2 = landmarks.landmark[8].x, landmarks.landmark[8].y
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist < threshold


def draw_lines(image, canvas, points):
    for i in range(1, len(points)):
        if points[i] is not None and points[i-1] is not None:
            cv2.line(image, points[i - 1], points[i], GREEN_RGB, 2)
            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 5)


def show_status(image, drawing_mode, prediction_done, predicted_class):
    if drawing_mode:
        cv2.putText(image, 'Press SPACE to predict | Press X to delete', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN_RGB, 2)
    elif prediction_done:
        if predicted_class is not None:
            text = f'You are drawing: {predicted_class}'
            color = GREEN_RGB
        else:
            text = f'The object drawn is too small!'
            color = RED_RGB
        cv2.putText(image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(image, 'Press SPACE to start drawing', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN_RGB, 2)
    


def handle_hand_landmarks(hand_landmarks, drawing_mode, points, pen_down, prev_pen_down):
    # Process the hand's landmark points and update the drawing point list
    if drawing_mode:
        pen_down = fingers_touching(hand_landmarks)
        x = int(hand_landmarks.landmark[8].x * 640)
        y = int(hand_landmarks.landmark[8].y * 480)
        if pen_down:
            if not prev_pen_down:
                points.append(None)
                points.append((x, y))
            else:
                points.append((x, y))
        prev_pen_down = pen_down
    return pen_down, prev_pen_down



def main(opt):
    cap = WebcamStream(0).start()
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    drawing_mode = False
    prediction_done = False
    predicted_class = None

    pen_down = False
    prev_pen_down = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuickDraw(20)
    model.to(device)
    model.load_state_dict(torch.load("trained_models2/best_checkpoint.pt", weights_only=True))
    model.eval()


    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            image = cap.read()
            if image is None:
                continue

            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                pen_down, prev_pen_down = handle_hand_landmarks(
                    hand_landmarks, drawing_mode, points, pen_down, prev_pen_down
                )
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            else:
                pen_down = False
                prev_pen_down = False

            draw_lines(image, canvas, points)
            show_status(image, drawing_mode, prediction_done, predicted_class)
            cv2.imshow('MediaPipe Hands', image)

            key = cv2.waitKey(5) & 0xFF

            if key == 27:
                break
            elif key == 32:
                if not drawing_mode:
                    drawing_mode = True
                    prediction_done = False
                    predicted_class = None
                    
                    points.clear()
                    canvas.fill(0)
                else:
                    drawing_mode = False
                    prediction_done = True
                    image = preprocess_image(canvas, opt.min_detect_area)
                    if image is not None:
                        image_tensor = image.to(device)
                        with torch.no_grad():
                            logits = model(image_tensor)
                            pred_idx = torch.argmax(logits[0]).item()
                            predicted_class = CLASSES[pred_idx]
                        
                    else:
                        predicted_class = None
                        
            elif key == ord('x') or key == ord('X'):
                points.clear()
                canvas.fill(0)
                drawing_mode = False
                prediction_done = False
                predicted_class = None
                pen_down = False
                prev_pen_down = False    

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = get_args()
    main(opt)
