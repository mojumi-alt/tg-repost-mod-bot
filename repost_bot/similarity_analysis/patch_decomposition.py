import numpy as np
import math


def image_to_array(img):
    return np.asarray(img)


def get_patches(array, patch_size, stride):

    patches = np.full(
        (
            int(math.ceil((array.shape[1] + 1 - patch_size) / stride))
            * int(math.ceil((array.shape[0] + 1 - patch_size) / stride)),
            patch_size,
            patch_size,
            array.shape[-1],
        ),
        0,
    )

    current = 0
    for y in range(
        0, stride * int(math.ceil((array.shape[1] + 1 - patch_size) / stride)), stride
    ):
        for x in range(
            0,
            stride * int(math.ceil((array.shape[0] + 1 - patch_size) / stride)),
            stride,
        ):
            patches[current] = array[x : (x + patch_size), y : (y + patch_size), ::]
            current += 1

    return patches
