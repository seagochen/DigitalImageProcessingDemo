import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from Utilities.DiagramPlotter import DiagramPlotter


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


def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def plot_gaussian_distribution():
    mu1, sig1 = 0, 1  # standard distribution
    mu2, sig2 = 1, 1  # move the chart to right
    mu3, sig3 = 0, 0.5  # increase the noise coverage
    mu4, sig4 = 0, 2.5  # increase the noise coverage
    x = np.linspace(-5, 5, 100)

    y1 = gaussian_noise_kernel(x, mu1, sig1)
    y2 = gaussian_noise_kernel(x, mu2, sig2)
    y3 = gaussian_noise_kernel(x, mu3, sig3)
    y4 = gaussian_noise_kernel(x, mu4, sig4)

    plt.plot(x, y1, 'r', label='mu1, sig1 = 0, 1')
    plt.plot(x, y2, 'g', label='mu2, sig2 = 1, 1')
    plt.plot(x, y3, 'b', label='mu3, sig3 = 0, 0.5')
    plt.plot(x, y4, 'm', label='mu4, sig4 = 0, 2.5')
    plt.legend()
    plt.grid()
    plt.show()

    # y1, y2, y3, y4 = [], [], [], []
    # for i in range(50):
    #     t1 = gaussian_noise_kernel(x1[i], mu1, sig1)
    #     t2 = gaussian_noise_kernel(x2[i], mu2, sig2)
    #     t3 = gaussian_noise_kernel(x3[i], mu3, sig3)
    #     t4 = gaussian_noise_kernel(x4[i], mu4, sig4)
    #
    #     y1.append(t1)
    #     y2.append(t2)
    #     y3.append(t3)
    #     y4.append(t4)


def plot_gaussian_noise():
    mu, sig = 0, 0.3  # standard distribution
    gx = np.linspace(mu - 5 * sig, mu + 5 * sig, 100)
    gaussian = gaussian_noise_kernel(gx, mu, sig)

    # line with gaussian noise
    noise = np.sin(gx)
    sine = np.sin(gx)

    for i in range(100):
        pos = int(100 * random.random())
        if gx[pos] == 0:  # nothing to change
            continue

        if gx[pos] < 0:
            noise[i] = noise[i] * (1 - gaussian[pos])
            continue

        if gx[pos] > 0:
            noise[i] = noise[i] * (1 + gaussian[pos])

    # plot gaussian distribution chart
    plt.plot(gx, gaussian, 'r', label='mu1, sig1 = 0, 1')
    plt.plot(gx, noise, 'm', label='noise')
    plt.plot(gx, sine, 'g', label='signal')
    plt.legend()
    plt.grid()
    plt.show()


def gaussian_noise(image, ratio, sigma):
    # generate gaussian kernel
    x = np.linspace(- 5, 5, 100)
    kernel = gaussian_noise_kernel(x, 0, sigma)

    # output image
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # apply gaussian noise
                pos = int(100 * random.random())
                if x[pos] == 0:  # nothing to change
                    output[i][j] = image[i][j]
                    continue

                if x[pos] < 0:
                    temp = image[i][j] * (1 - kernel[pos])
                    if temp < 0:
                        output[i][j] = 0
                    else:
                        output[i][j] = temp
                    continue

                if x[pos] > 0:
                    temp = image[i][j] * (1 + kernel[pos])
                    if temp > 255:
                        output[i][j] = 255
                    else:
                        output[i][j] = temp

            else:
                output[i][j] = image[i][j]

    return output


def salt_pepper_demo():
    img = cv2.imread("Data/Illustrations/kana.jpg", cv2.IMREAD_GRAYSCALE)

    # add salt pepper noise
    spn_30_img = salt_pepper_noise(img, .30)  # 30%
    spn_50_img = salt_pepper_noise(img, .50)  # 50%
    spn_90_img = salt_pepper_noise(img, .90)  # 90%

    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(spn_30_img, "salt pepper 30%")
    plt.append_image(spn_50_img, "salt pepper 50%")
    plt.append_image(spn_90_img, "salt pepper 90%")
    plt.show(2, 2)


def gaussian_demo():
    img = cv2.imread("Data/Illustrations/kana.jpg", cv2.IMREAD_GRAYSCALE)

    # add gaussian noise
    gass_30_img = gaussian_noise(img, .30, 1)  # 30%
    gass_50_img = gaussian_noise(img, .50, 1)  # 50%
    gass_90_img = gaussian_noise(img, .50, 0.5)  # 90%

    plt = DiagramPlotter()
    plt.append_image(img, "original")
    plt.append_image(gass_30_img, "gaussian 30%, sig=1 mu=0")
    plt.append_image(gass_50_img, "gaussian 50%, sig=1 mu=0")
    plt.append_image(gass_90_img, "gaussian 50%, sig=0.5 mu=0")
    plt.show(2, 2)


def noise_generator_test():
    salt_pepper_demo()
    gaussian_demo()
    plot_gaussian_noise()
    plot_gaussian_distribution()
