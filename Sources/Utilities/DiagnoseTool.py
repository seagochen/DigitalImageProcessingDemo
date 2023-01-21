import cv2
import numpy as np


def calculate_hist(img):
    histogram = np.zeros(256, dtype=np.uint64)

    row, col = img.shape
    for r in range(row):
        for c in range(col):
            histogram[img[r, c]] = histogram[img[r, c]] + 1

    return histogram


def load_image_gray(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return image


def load_image_color(file: str):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    return image
