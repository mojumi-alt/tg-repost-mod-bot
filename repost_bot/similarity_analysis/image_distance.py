import logging
from numpy.core.fromnumeric import argmin
from .patch_decomposition import get_patches, image_to_array
import numpy as np
import time


def compute_image_correlation(left, right, patch_size, stride):

    left = image_to_array(left)
    right = image_to_array(right)
    left_patches = get_patches(left, patch_size, stride)
    right_patches = get_patches(right, patch_size, stride)

    return compute_patch_set_correlation(left_patches, right_patches)


def compute_patch_set_correlation(left, right):

    left_sum = (1 / right.shape[0]) * sum(
        min(np.sum(np.abs(left - r), axis=(1, 2, 3))) for r in right
    )

    right_sum = (1 / left.shape[0]) * sum(
        min(np.sum(np.abs(right - l), axis=(1, 2, 3))) for l in left
    )
    return left_sum + right_sum


def compute_image_set_correspondence(search_set, reference_set):

    results = []

    for l in search_set:
        temp = [compute_patch_set_correlation(l, r) for r in reference_set]
        minimum = argmin(temp)
        results.append((minimum, temp[minimum]))

    return results
