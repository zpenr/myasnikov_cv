import cv2
import numpy as np
import time
from math import dist
import json
from pathlib import Path
from enum import Enum
from itertools import permutations
from random import choice

save_path = Path(__file__).parent / "positions.json"

cap = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

class Color(Enum):
    YELLOW = 1
    RED = 2
    BLUE = 3
    GREEN = 4

position = [0,0]
clicked = False

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position, clicked
        position = [x, y]
        clicked = True
cv2.setMouseCallback("Image", on_click)

lowers = {Color.YELLOW: None, Color.RED: None, Color.BLUE: None, Color.GREEN: None}
uppers = {Color.YELLOW: None, Color.RED: None, Color.BLUE: None, Color.GREEN: None}

if save_path.exists():
    with open(save_path, "r") as f:
        data = json.load(f)
        lowers = {
            Color.YELLOW: np.array(data["yellow"]["lower"], dtype="uint8") if data["yellow"]["lower"] is not None else None,
            Color.RED: np.array(data["red"]["lower"], dtype="uint8") if data["red"]["lower"] is not None else None,
            Color.BLUE: np.array(data["blue"]["lower"], dtype="uint8") if data["blue"]["lower"] is not None else None,
            Color.GREEN: np.array(data["green"]["lower"], dtype="uint8") if data["green"]["lower"] is not None else None
        }
        uppers = {
            Color.YELLOW: np.array(data["yellow"]["upper"], dtype="uint8") if data["yellow"]["upper"] is not None else None,
            Color.RED: np.array(data["red"]["upper"], dtype="uint8") if data["red"]["upper"] is not None else None,
            Color.BLUE: np.array(data["blue"]["upper"], dtype="uint8") if data["blue"]["upper"] is not None else None,
            Color.GREEN: np.array(data["green"]["upper"], dtype="uint8") if data["green"]["upper"] is not None else None
        }

positions = []


perm = list(permutations([Color.YELLOW, Color.RED, Color.BLUE, Color.GREEN]))
sequence_colors = choice(perm)
print("Sequence:", [color.name for color in sequence_colors][::-1])
curr_color = Color.YELLOW
positions = {Color.YELLOW: None, Color.RED: None, Color.BLUE: None, Color.GREEN: None}
while True:

    ret, frame = cap.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    if not ret:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('1'):
        curr_color = Color.YELLOW
    if cv2.waitKey(1) & 0xFF == ord('2'):
        curr_color = Color.RED
    if cv2.waitKey(1) & 0xFF == ord('3'):
        curr_color = Color.BLUE
    if cv2.waitKey(1) & 0xFF == ord('4'):
        curr_color = Color.GREEN

    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        lowers[curr_color] = np.clip(color * 0.8, 0, 255).astype("uint8")
        uppers[curr_color] = np.clip(color * 1.2, 0, 255).astype("uint8")
        uppers[curr_color][1] = 255
        uppers[curr_color][2] = 255
   
    for color in Color:
        if lowers[color] is not None and uppers[color] is not None:
            inr = cv2.inRange(hsv, lowers[color], uppers[color])
            mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                (x,y), r = cv2.minEnclosingCircle(largest_contour)
                if color == Color.RED:
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 2)
                if color == Color.YELLOW:
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)
                if color == Color.BLUE:
                    cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 0), 2)
                if color == Color.GREEN:
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                positions[color] = (int(x), int(y))

    if positions[sequence_colors[0]] is not None and positions[sequence_colors[1]] is not None and positions[sequence_colors[2]] is not None and positions[sequence_colors[3]] is not None:
        if positions[sequence_colors[0]][0] < positions[sequence_colors[1]][0] < positions[sequence_colors[2]][0] < positions[sequence_colors[3]][0]:
            cv2.putText(frame, "Done", (40,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
            print("Done")
    cv2.putText(frame, f"Current color: {curr_color.name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Image', frame)

cap.release()
cv2.destroyAllWindows()

with open(save_path, "w") as f:
    json.dump({"red": {"lower": lowers[Color.RED].tolist() if lowers[Color.RED] is not None else None,
                       "upper": uppers[Color.RED].tolist() if uppers[Color.RED] is not None else None},
                "yellow": {"lower": lowers[Color.YELLOW].tolist() if lowers[Color.YELLOW] is not None else None,
                           "upper": uppers[Color.YELLOW].tolist() if uppers[Color.YELLOW] is not None else None},
                "blue": {"lower": lowers[Color.BLUE].tolist() if lowers[Color.BLUE] is not None else None,
                          "upper": uppers[Color.BLUE].tolist() if uppers[Color.BLUE] is not None else None},
                "green": {"lower": lowers[Color.GREEN].tolist() if lowers[Color.GREEN] is not None else None,
                          "upper": uppers[Color.GREEN].tolist() if uppers[Color.GREEN] is not None else None}},f
              )