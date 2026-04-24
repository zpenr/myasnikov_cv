import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label,regionprops
from skimage.io import imread
from skimage.color import rgb2hsv
image = imread('balls_and_rects.png')

hsv = rgb2hsv(image)
h = hsv[:,:,0]

colors_rect = []
colors_crcl = []
for color in np.unique(h):
    if color == 0:
        continue
    binary = h == color
    labeled = label(binary)
    props = regionprops(labeled)
    
    for prop in props:
        norm_area = prop.area_bbox / prop.area
        if norm_area == 1:
            colors_rect.append(color)
        elif norm_area > 0.75:
            colors_crcl.append(color)


groups_rect = [[colors_rect[0]]]
groups_crcl = [[colors_crcl[0]]]
delta = 0.05
cgrp = 0

for i in range(1, len(colors_rect)):
    if abs(colors_rect[i-1] - colors_rect[i]) < delta:
        groups_rect[-1].append(colors_rect[i])
    else:
        groups_rect.append([])

for i in range(1, len(colors_crcl)):
    if abs(colors_crcl[i-1] - colors_crcl[i]) < delta:
        groups_crcl[-1].append(colors_crcl[i])
    else:
        groups_crcl.append([])

print(f"Всего: {len(colors_crcl) + len(colors_rect)}")
print(f"Многоугольники: {len(colors_rect)}")
for grp in groups_rect:
    avg_color = np.mean(grp)
    print(avg_color, len(grp))

print(f"Круги: {len(colors_crcl)}")

for grp in groups_crcl:
    avg_color = np.mean(grp)
    print(avg_color, len(grp))

plt.subplot(121)
plt.imshow(h, cmap="gray")
plt.subplot(122)
plt.plot(np.unique(h), 'o')
plt.show()