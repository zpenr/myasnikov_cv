import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure
from scipy.optimize import linear_sum_assignment

path = Path('out')
files = sorted(path.glob('*.npy'))

def get_center(image: np.ndarray, label: int):
    center = np.argwhere(image == label).mean(axis=0)
    return center

def get_all_centers(file_path: Path | str) -> np.ndarray[np.ndarray]:
    data = np.load(file_path)
    labeled = measure.label(data)
    centers = []
    for label in sorted(np.unique(labeled))[1:]:
        center = get_center(labeled, label)
        centers.append(center)
    return np.array(centers)

def dist(center1: list[int, int], center2: list[int,int]) -> int:
    return ((center1[1] - center2[1])**2 + (center1[0] - center2[0])**2)**0.5

def get_distances(pre_centers: np.ndarray[np.ndarray], cur_centers: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    distances = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            distances[i,j] = dist(pre_centers[i], cur_centers[j])
    return distances

traectories = [[], [], []]
prev_positons = None
speeds = np.zeros((3, 2))


for i, file_path in enumerate(files):
    curr_positions = get_all_centers(file_path)
    
    if i == 0:
        prev_positons = curr_positions
        for j in range(3):
            traectories[j].append(prev_positons[j])

    predicted_pos = prev_positons + speeds
    
    dist_matrix = get_distances(predicted_pos, curr_positions)

    old_labels, new_labels = linear_sum_assignment(dist_matrix)

    new_centers_ordered = np.zeros((3, 2))

    for old_label, new_label in zip(old_labels, new_labels):
        found_pos = curr_positions[new_label]
        
        d = dist(found_pos, prev_positons[old_label])
        
        if d > 35 and i > 5:
            new_pos = prev_positons[old_label] + speeds[old_label]
        else:
            new_pos = found_pos
        new_speed = new_pos - prev_positons[old_label]
        speeds[old_label] = new_speed

        traectories[old_label].append(new_pos)
        new_centers_ordered[old_label] = new_pos
        
    prev_positons = new_centers_ordered

plt.figure(figsize=(10, 10))
colors = ['red', 'blue', 'green']

for j in range(3):
    traectory = np.array(traectories[j])
    plt.plot(traectory[:, 1], traectory[:, 0], color=colors[j], 
            marker='o', markersize=2, alpha=0.7,
            linewidth=1, label=f'Ball {j}')

plt.gca().invert_yaxis()
plt.legend()
plt.show()