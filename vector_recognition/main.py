import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1]+2))
    new_image[1:-1,1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image, 0) / shape[0] == 1).sum()
    hlines = (np.sum(image, 1) / shape[1] == 1).sum()
    return vlines, hlines

def w(regions, transponse = False) -> float:
    """
    Vadim and Vitalya
    """
    shape = region.image.shape
    image = region.image
    if transponse:
        image = image.T
    top = image[:shape[0] // 2]
    bottom = image[-(shape[0] // 2):]
    bottom = bottom[::-1]
    result = bottom == top
    return result.sum() / result.size

def exctractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    vlines, hlines = count_lines(region)
    vlines /= region.image.shape[1]
    hlines /= region.image.shape[0]
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]
    vitalya_x = w(region, transponse=True)
    vitalya_y = w(region)
    return np.array([region.area/region.image.size,
                    cy, cx, perimeter, 
                    holes,
                    vlines, hlines, 
                    eccentricity, aspect,
                    vitalya_x, vitalya_y])

def classificator(region, templates: dict):
    prop = exctractor(region)
    result = ""
    min_d = 10 ** 16
    for symbol, t in templates.items():
        d = ((t-prop)**2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result


if __name__ == "__main__":
    template = imread('alphabet-small.png')[:,:,:-1]

    template = template.sum(2)
    binary = template != 765

    labeled = label(binary)
    props = regionprops(labeled)

    templates = {}

    for region, symbol in zip(props, ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]):
        templates[symbol] = exctractor(region)

    print(classificator(props[0], templates))

    image = imread('alphabet.png')[:,:,:-1]
    abinary = image.mean(2) > 0
    alabeled = label(abinary)

    aprops = regionprops(alabeled)
    result = {}

    image_path = save_path / "out"
    image_path.mkdir(exist_ok=True)

    plt.figure(figsize=(5,7))

    for region in aprops:
        symbol = classificator(region, templates)
        if symbol not in result:
            result[symbol] = 0
        result[symbol]+=1
        plt.cla()
        plt.title(f"Class - '{symbol}'")
        plt.imshow(region.image)
        plt.savefig(image_path / f"image_{region.label}.png")
    print(result)

    # plt.imshow(binary_alphabe)
    # plt.show()