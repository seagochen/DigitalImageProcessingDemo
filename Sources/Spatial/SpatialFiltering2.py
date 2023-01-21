import cv2
import numpy as np

from Utilities.DiagramPlotter import DiagramPlotter


def laplacian_kernel(image, i, j):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    sub_img = image[i - 1:i + 2, j - 1:j + 2]
    result = kernel * sub_img
    return abs(np.sum(result))


def laplacian_operator(image):
    width, height = image.shape
    backup = np.zeros(image.shape, np.uint8)

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            backup[i][j] = laplacian_kernel(image, i, j)

    return backup


def laplacian_op_test(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    output = laplacian_operator(img)

    # plots images
    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(output, "laplacian operator")
    plt.show(1, 2)


def laplacian_kernel_mod(image, i, j, k):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # 从原始图像中，根据ij所在位置，扣出一个3x3的小矩阵
    sub_img = image[i - 1:i + 2, j - 1:j + 2]
    result = kernel * sub_img
    result = abs(np.sum(result)) * k
    return round(result)


def laplacian_operator_mod(image, k):
    width, height = image.shape
    backup = np.zeros(image.shape, np.uint8)

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            res = laplacian_kernel_mod(image, i, j, k) + image[i][j]

            if res > 255:
                res = 255

            backup[i][j] = res

    return backup


def laplacian_op_test():
    img = cv2.imread("Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0338(a)(blurry_moon).tif", cv2.IMREAD_GRAYSCALE)
    output = laplacian_operator_mod(img, 2.5)

    # plots images
    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(output, "enhanced")
    plt.show(1, 2)
