import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from Utilities.DiagramPlotter import DiagramPlotter


def poisson_distribution(lam: float, limits: int):
    distributions = []

    e_lam = math.pow(math.e, -lam)
    sum = 0

    for k in range(0, 100):
        lambda_k = math.pow(lam, k)
        k_factorial = math.factorial(k)
        prob = (lambda_k / k_factorial) * e_lam
        sum = sum + prob

        if limits - 1 <= k:
            others = 1 - sum
            distributions.append(others)
            break

        if prob > 0:
            distributions.append(prob)
            k = k + 1
        else:
            break

    return distributions


def generate_poisson_list(distribution, size=100):
    pos = 0
    poisson_list = []

    for prob in distribution:
        max = round(size * prob)
        p_list = []

        if max < 1:
            break

        for i in range(max):
            p_list.append(pos)

        pos = pos + 1
        poisson_list.append(p_list)

    poisson_list = [i for elem in poisson_list for i in elem]
    return poisson_list


def poisson_value1(poisson_list):
    number = random.random() * (len(poisson_list) - 1)
    number = round(number)
    try:
        return poisson_list[number]
    except:
        print(number, len(poisson_list))


def poisson_value2(lamb):
    L = math.exp(-lamb)
    k = 0
    p = 1

    while p >= L:
        k = k + 1
        # Generate uniform random number u in [0, 1] and let p ← p × u.
        p = random.random() * p

    return k - 1


def poisson_noise1(image, poisson_list, dts, lamb):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # get the poisson value
            noise_value = 0
            for k in range(dts):
                noise_value = noise_value + poisson_value1(poisson_list)

            # add noise to original image
            temp = image[i][j] + noise_value - dts * lamb
            if temp > 255:
                temp = 255
            if temp < 0:
                temp = 0

            # assign noised image to output
            output[i][j] = temp

    return output


def poisson_noise2(image, lamb):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # get the poisson value
            noise_value = 0
            for k in range(image[i][j]):
                noise_value = noise_value + poisson_value2(lamb)

            # add noise to original image
            temp = noise_value
            if temp > 255:
                temp = 255

            # assign noised image to output
            output[i][j] = temp

    return output


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


def gamma_transform(file: str, c, p_list, amplify,
                    g1, t1,
                    g2, t2,
                    g3, t3):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # # normalization the pixels
    float_image = image / 255.0

    # gamma 1
    gamma1 = c * np.power(float_image, g1)
    gamma1 = np.uint8(gamma1 * 255.0)
    gamma1 = poisson_noise2(gamma1)

    # gamma 2
    gamma2 = c * np.power(float_image, g2)
    gamma2 = np.uint8(gamma2 * 255.0)
    gamma2 = poisson_noise2(gamma2)

    # gamma 3
    gamma3 = c * np.power(float_image, g3)
    gamma3 = np.uint8(gamma3 * 255.0)
    gamma3 = poisson_noise2(gamma3)

    # show images
    show_gamma_images(image, gamma1, t1, gamma2, t2, gamma3, t3)


def poisson_noise_test():
    # read image from file
    img = cv2.imread("Data/Illustrations/kana.jpg", cv2.IMREAD_GRAYSCALE)

    # poisson noise image with lambda is 1.5
    # d = poisson_distribution(1.5, 100)
    # p_list = generate_poisson_list(d, 1000)
    # lam_15 = poisson_noise1(img, p_list, 100,  1.5)
    # lam_15 = poisson_noise2(img, 1)

    # poisson noise image with lambda is 2.5
    d = poisson_distribution(1.5, 1000)
    p_list = generate_poisson_list(d, 1000)
    lam_12 = poisson_noise1(img, p_list, 50, 1.5)
    # lam_25 = poisson_noise2(img, 1.2)

    # poisson noise image with lambda is 3.5
    # d = poisson_distribution(3.5, 100)
    # p_list = generate_poisson_list(d, 1000)
    # lam_35 = poisson_noise1(img, p_list, 100, 3.5)
    # lam_35 = poisson_noise2(img, 3.5)

    # plot images
    pt = DiagramPlotter()
    pt.append_image(img, "original")
    # pt.append_image(lam_25, "lambda 1.5")
    pt.append_image(lam_12, "lambda 1.2")
    # pt.append_image(lam_35, "lambda 3.5")
    pt.show(1, 2)
