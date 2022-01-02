import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
import math
import numpy as np
from PIL import Image


def show_patch(patch):
    plt.imshow(patch)
    plt.show()


def show_patches(patches):

    side_length = int(math.ceil(math.sqrt(patches.shape[0])))
    patch_size = patches.shape[1]

    complete_image = np.full((side_length * patch_size, side_length * patch_size, 3), 0)

    patch_iter = iter(patches)
    for y in range(0, side_length * patch_size, patch_size):
        for x in range(0, side_length * patch_size, patch_size):
            try:
                current_patch = next(patch_iter)
            except StopIteration:
                break
            for i in range(patch_size):
                for j in range(patch_size):
                    for c in range(patches.shape[3]):
                        complete_image[x + i, y + j, c] = current_patch[i, j, c]

    plt.imshow(complete_image)
    plt.show()


def show_image(img):

    plt.imshow(img)
    plt.show()


def visualize_nn_map(search_set, reference_set, nn_map):

    search_images = [Image.open(l) for l in search_set]
    target_images = [Image.open(r) for r in reference_set]

    for i in range(len(search_images)):

        fig, (l, r) = plt.subplots(1, 2)

        l.imshow(search_images[i])
        r.imshow(target_images[nn_map[i]])
        plt.show()
