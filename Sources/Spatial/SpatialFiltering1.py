import random

import cv2
import numpy as np

from Utilities.DiagramPlotter import DiagramPlotter


def adjustable_mean_kernel(image, size, i, j):
    if size % 2 == 0:
        size = size - 1

    sum = 0
    mid = int((size - 1) / 2)

    for subtile_x in range(size):
        for subtile_y in range(size):
            sum = sum + int(image[i + subtile_x - mid][j + subtile_y - mid])

    sqrt = size * size
    output = round(sum / sqrt)
    return output


def mean_kernel(image, size=3):
    width, height = image.shape
    backup = np.zeros(image.shape, np.uint8)

    mid = int((size - 1) / 2)

    for i in range(mid, width - mid):
        for j in range(mid, height - mid):
            backup[i][j] = adjustable_mean_kernel(image, size, i, j)

    return backup


def mean_filter_test(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    output = mean_kernel(img, 5)

    # plots images
    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(output, "mean kernel")
    plt.show(1, 2)


def adjustable_median_kernel(image, size, i, j):
    if size % 2 == 0:
        size = size - 1

    mid = int((size - 1) / 2)
    numbs_list = []

    for subtile_x in range(size):
        for subtile_y in range(size):
            numbs_list.append(image[i + subtile_x - mid][j + subtile_y - mid])

    if len(numbs_list) % 2 == 0:
        numbs_list = np.sort(numbs_list)
        n1 = numbs_list[int(len(numbs_list) / 2)]
        n2 = numbs_list[int(len(numbs_list) / 2 - 1)]

        return int((n1 + n2) / 2)
    else:
        numbs_list = np.sort(numbs_list)
        return numbs_list[int(len(numbs_list) / 2)]


def median_kernel(image, size=3):
    width, height = image.shape
    backup = np.zeros(image.shape, np.uint8)

    mid = int((size - 1) / 2)

    for i in range(mid, width - mid):
        for j in range(mid, height - mid):
            backup[i][j] = adjustable_median_kernel(image, size, i, j)

    return backup


def salt_pepper_noise(image, ratio):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # salt pepper noise
                if random.random() > 0.5:  # change the pixel to 255
                    output[i][j] = 255
                else:
                    output[i][j] = 0
            else:
                output[i][j] = image[i][j]

    return output


def median_filter_test():
    img = cv2.imread("Data/Illustrations/kana.jpg", cv2.IMREAD_GRAYSCALE)
    sp_img = salt_pepper_noise(img, .05)
    output = median_kernel(sp_img, 5)

    # plots images
    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(sp_img, "salt pepper, 5%")
    plt.append_image(output, "median kernel")
    plt.show(1, 3)
