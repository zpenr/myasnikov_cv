import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import morphology


image = np.load(f'wires2.npy')
struct = np.ones((3,1))
processed = morphology.opening(image, footprint=struct) 
labeled = label(image)
wires = labeled.max()
for i in range(1,wires + 1):  
    print( f' Wires = {i} , parts = {label(((labeled == i)*processed)).max()}')
plt.imshow(processed)
plt.show()