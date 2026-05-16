from pynput.mouse import Listener
import pynput
import numpy as np
import cv2
import time
import mss
import threading
from enum import Enum
import math
import pathlib
bbox = []
path = pathlib.Path(__file__)
class State(Enum):
    JUMP = "jump"
    DOWN = "down"
    FLY_DOWN = "fly"


def off_jump():
    global state
    state = State.DOWN

def on_jump():
    global state
    state = State.JUMP

def down():
    keyboard.press(down_key)
    global state
    state = State.FLY_DOWN
    

def jump_and_duck(x, w, elapsed, speed):
    on_jump()
    keyboard.press(space_key)
    hold_time = 0.1
    threading.Timer(hold_time, lambda: keyboard.release(space_key)).start()
    if w > 100:
        sleep_time = (x + w) / speed
    else:
        sleep_time = (x + w) / speed
    threading.Timer(sleep_time, lambda: down()).start()

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if pressed:
        global start_time
        start_time = time.time()
        bbox.append((x,y))
    if len(bbox) == 2:
        return False
    
with Listener(on_click=on_click) as listener:
    listener.join()

keyboard = pynput.keyboard.Controller()
template = cv2.imread(path / "template.png", cv2.IMREAD_GRAYSCALE)
template = cv2.threshold(template, 90, 255, cv2.THRESH_BINARY_INV)[1]
template_h, template_w = template.shape
space_key = pynput.keyboard.Key.space
down_key = pynput.keyboard.Key.down

state = State.DOWN

print("time")
start_time = time.time()
cv2.namedWindow("Game", cv2.WINDOW_GUI_NORMAL)


sct = mss.mss()
monitor = {"top": bbox[0][1], "left": bbox[0][0], 
           "width": bbox[1][0] - bbox[0][0], 
           "height": bbox[1][1] - bbox[0][1]}

def get_current_speed(elapsed):
    speed = 360 + 1.6125 * elapsed
    return min(780, speed)

# def get_current_speed(t):
#     return 1030 - 800 * math.exp(-t / 58)

image = np.array(sct.grab(monitor))
gray_img = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
_, binary = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY_INV)

result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
width = binary.shape[1]
game_ground_y = top_left[1] + template_h
prev_min_x = 10_000
min_x = 10_000
prev_time = time.time()
current_speed = 0
while True:
    image = np.array(sct.grab(monitor))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY_INV)

    result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cutted = opened[:, top_left[0] + template_w:]
    contours = cv2.findContours(cutted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    elapsed = time.time() - start_time
    max_x = 55 + 1.5 * elapsed
    t_react = 0.25
    current_speed = get_current_speed(elapsed)
    max_x = current_speed * t_react

    rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:
            x, y, w, h = cv2.boundingRect(contour)
            rects.append((x, y, w, h))

    if rects:
        rects.sort(key=lambda r: r[0])
        min_x = rects[0][0]
        merged = []
        cur = list(rects[0])
        for r in rects[1:]:
            if r[0] - (cur[0] + cur[2]) < 20:
                new_x = min(cur[0], r[0])
                new_y = min(cur[1], r[1])
                new_w = max(cur[0] + cur[2], r[0] + r[2]) - new_x
                new_h = max(cur[1] + cur[3], r[1] + r[3]) - new_y
                cur = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(cur))
                cur = list(r)
        merged.append(tuple(cur))
        for (x, y, w, h) in merged:
            if 5 < x < max_x and state == State.DOWN:
                ground_y = top_left[1] + template_h
                if y + h < ground_y - template_h * 0.8 or y  > ground_y:
                    pass
                elif y + h < game_ground_y - template_h * 0.6:   

                    keyboard.press(down_key)
                    threading.Timer((x+w+70)/current_speed, lambda: keyboard.release(down_key)).start()
                else:
                    jump_and_duck(x, w, elapsed,current_speed)
                break

            cv2.rectangle(image, (x + top_left[0] + template_w, y), 
                          (x + top_left[0] + template_w + w, y + h), (0, 0, 255), 2)
        if state == State.FLY_DOWN and game_ground_y-10<top_left[1]+template_h<game_ground_y + 45:
            keyboard.release(down_key)
            state = State.DOWN

    cv2.rectangle(image, top_left, (top_left[0] + template_w, top_left[1] + template_h), (0, 255, 0), 2)
    cv2.putText(image,state.value,(20,20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,40),2)
    cv2.imshow("Game", image)
    prev_time = time.time()
    prev_min_x = min_x
    print(current_speed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break