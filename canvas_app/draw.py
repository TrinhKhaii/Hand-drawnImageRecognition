"""
@author: Trinh Khai Truong 
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from canvas_app.GameState import GameState
from src.constants import *
from src.model import QuickDraw
from src.utils import preprocess_image

def get_args():
    parser = argparse.ArgumentParser(description="Hand-drawn Image Recognition")
    parser.add_argument("--min-detect-area", type=int, default=3000, help="Minimum area (in pixels) of the detected pointer object to be considered valid")
    parser.add_argument("--prediction-display-time", type=int, default=3, help="Duration (in seconds) to display the prediction result on the screen")
    return parser.parse_args()


def paint_draw(event, x, y, flags, state: GameState):
    if not state.drawing_started:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.ix, state.iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        cv2.line(state.image, (state.ix, state.iy), (x, y), WHITE_RGB, THICKNESS)
        state.ix, state.iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        cv2.line(state.image, (state.ix, state.iy), (x, y), WHITE_RGB, THICKNESS)
        state.ix, state.iy = x, y


def predict(model, device, image, area_threshold):
    img_tensor = preprocess_image(image, area_threshold)
    if img_tensor is None:
        return None, 0.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
        if 0 <= pred_idx < len(CLASSES):
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0, pred_idx].item() * 100
            return CLASSES[pred_idx], confidence
    return None, 0.0


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuickDraw(num_classes=len(CLASSES))
    model.load_state_dict(torch.load('trained_models2/best_checkpoint.pt', weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    state = GameState(CLASSES)
    cv2.namedWindow("QuickDraw")
    cv2.setMouseCallback('QuickDraw', lambda e, x, y, f, p: paint_draw(e, x, y, f, state))

    while True:
        display_img = 255 - state.image.copy()

        if not state.game_started:
            cv2.putText(display_img, "Press SPACE to start!", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE_RGB, 2)
        else:
            if not state.drawing_started:
                cv2.putText(display_img, f"Draw: {state.current_class} | Press SPACE to start!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_RGB, 2)
            else:
                info_text = f"Draw: {state.current_class}"
                if state.predicted_class:
                    info_text += f" | Predict: {state.predicted_class} ({state.predicted_confidence:.2f}%)"
                if state.start_time:
                    elapsed = time.time() - state.start_time
                    remaining = max(0, state.max_time - elapsed)
                    info_text += f" | Time remaining: {int(remaining)}s"

                cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_RGB, 2)
                cv2.putText(display_img, "Press SPACE to predict | Press X to delete", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_RGB, 2)

        cv2.imshow('QuickDraw', display_img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
        elif k == 32:
            if not state.game_started:
                state.game_started = True
                state.current_class = state.classes_to_draw[state.current_round]
            elif not state.drawing_started:
                state.drawing_started = True
                state.start_time = time.time()
                state.image[:] = 0
                state.predicted_class = ""
                state.predicted_confidence = 0.0
            else:
                pred_class, confidence = predict(model, device, state.image, opt.min_detect_area)
                if pred_class is not None:
                    state.predicted_class = pred_class
                    state.predicted_confidence = confidence
                    color = GREEN_RGB if pred_class == state.current_class else RED_RGB
                    text = (f"Correct! Predict: {pred_class} ({confidence:.2f}%)" if pred_class == state.current_class
                            else f"Wrong! Predict: {pred_class} ({confidence:.2f}%)")
                else:
                    state.predicted_class = "Invalid prediction"
                    state.predicted_confidence = 0.0
                    color = RED_RGB
                    text = "Invalid prediction"

                result_img = 255 - state.image.copy()
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow('QuickDraw', result_img)
                cv2.waitKey(opt.prediction_display_time * 1000)

                state.user_drawings.append(state.image.copy())
                state.current_round += 1
                if state.current_round >= state.max_rounds:
                    break
                state.current_class = state.classes_to_draw[state.current_round]
                state.drawing_started = False
                state.start_time = None
                state.image[:] = 0
                state.predicted_class = ""
                state.predicted_confidence = 0.0

        elif k == ord('x') or k == ord('X'):
            if state.drawing_started:
                state.image[:] = 0
                state.predicted_class = ""
                state.predicted_confidence = 0.0

        if state.start_time and state.drawing_started:
            elapsed = time.time() - state.start_time
            if elapsed > state.max_time:
                pred_class, confidence = predict(model, device, state.image, opt.min_detect_area)
                if pred_class is not None:
                    state.predicted_class = pred_class
                    state.predicted_confidence = confidence
                else:
                    state.predicted_class = "Invalid prediction"
                    state.predicted_confidence = 0.0

                state.user_drawings.append(state.image.copy())

                result_img = 255 - state.image.copy()
                result_text = f"Time's up! Prediction: {state.predicted_class}"
                
                cv2.putText(result_img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED_RGB, 2)
                cv2.imshow('QuickDraw', result_img)
                cv2.waitKey(opt.prediction_display_time * 1000)

                state.current_round += 1
                if state.current_round >= state.max_rounds:
                    break

                state.current_class = state.classes_to_draw[state.current_round]
                state.drawing_started = False
                state.start_time = None
                state.image[:] = 0
                state.predicted_class = ""
                state.predicted_confidence = 0.0

    cv2.destroyAllWindows()

    if len(state.user_drawings) > 0:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()
        for i, drawing_img in enumerate(state.user_drawings):
            axs[i].imshow(cv2.cvtColor(255 - drawing_img, cv2.COLOR_BGR2RGB))
            axs[i].set_title(f"Draw {i + 1}: {state.classes_to_draw[i]}")
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
