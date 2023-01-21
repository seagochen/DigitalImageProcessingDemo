import cv2
import matplotlib.pyplot as plt
import numpy as np


def piecewise_transformation(image):
    row, col, shape = image.shape
    out_img = np.zeros((row, col))

    # image conversion
    for r in range(row):
        for l in range(col):
            val = image[r, l, 0]
            if val < 90:
                out_img[r, l] = val * 0.25
            elif 90 <= val < 160:
                out_img[r, l] = val * 1.25
            else:
                out_img[r, l] = val * 0.25

    return out_img


def show_images(image1, image2):

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap="gray")
    plt.axis('off')
    plt.title("Original Image")

    # gamma 1
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap="gray")
    plt.axis('off')
    plt.title("Image Conversion")

    # show plots
    plt.show()


def intensity_test_2():
    # load image from file
    org_image = cv2.imread("Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0316(4)(bottom_left).tif")

    # piecewise conversion
    con_image = piecewise_transformation(org_image)
    show_images(org_image, con_image)
    # show images

