import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_images(original, data, title: str):

    # original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.axis('off')
    plt.title("Original Image")

    # image on the middle
    plt.subplot(1, 2, 2)
    plt.imshow(data, cmap="gray")
    plt.axis('off')
    plt.title(title)

    # show plots
    plt.show()


def show_gamma_images(img, gamma1, t1,
                      gamma2, t2,
                      gamma3, t3):
    # original
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.title("Original Image")

    # gamma 1
    plt.subplot(2, 2, 2)
    plt.imshow(gamma1, cmap="gray")
    plt.axis('off')
    plt.title(t1)

    # gamma 2
    plt.subplot(2, 2, 3)
    plt.imshow(gamma2, cmap="gray")
    plt.axis('off')
    plt.title(t2)

    # gamma 3
    plt.subplot(2, 2, 4)
    plt.imshow(gamma3, cmap="gray")
    plt.axis('off')
    plt.title(t3)

    # show plots
    plt.show()


def image_negative(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    negative_img = 255 - image.flatten()
    negative_img = negative_img.reshape(image.shape)
    show_images(image, negative_img, "Negative Image")


def log_transform(file: str):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    log_trans = 10 * np.log(1 + image)
    log_trans = np.uint8(log_trans)
    show_images(image, log_trans, "Log Transform")


def gamma_transform(file: str, c,
                    g1, t1,
                    g2, t2,
                    g3, t3):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # # normalization the pixels
    float_image = image / 255.0

    gamma1 = c * np.power(float_image, g1)
    gamma1 = np.uint8(gamma1 * 255.0)

    gamma2 = c * np.power(float_image, g2)
    gamma2 = np.uint8(gamma2 * 255.0)

    gamma3 = c * np.power(float_image, g3)
    gamma3 = np.uint8(gamma3 * 255.0)

    # show images
    show_gamma_images(image, gamma1, t1, gamma2, t2, gamma3, t3)


def intensity_test_1():
    # image negative
    image_negative("Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0304(a)(breast_digital_Xray).tif")

    # log transform
    log_transform("Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0305(a)(DFT_no_log).tif")

    # gamma transform
    gamma_transform(
        "Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0308(a)(fractured_spine).tif",
        1,
        0.6, "gamma 0.6",
        0.4, "gamma 0.4",
        0.3, "gamma 0.3"
    )
    gamma_transform(
        "Data/DIP/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0309(a)(washed_out_aerial_image).tif",
        1,
        3, "gamma 3",
        4, "gamma 4",
        5, "gamma 5"
    )

    pass
