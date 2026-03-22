import numpy as np
from skimage.measure import label
from skimage import morphology

image = np.load('stars.npy')

plus_struct = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]])

cross_struct = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]])

plus_image = morphology.opening(image, footprint=plus_struct) 
cross_image = morphology.opening(image, footprint=cross_struct)

plus_labeled = label(plus_image)
cross_labeled = label(cross_image)

num = cross_labeled.max() + plus_labeled.max()
print(num)