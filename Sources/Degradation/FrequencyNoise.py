from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter

import cv2
import numpy as np


# def frequency_noise_analysis(filepath: str):
#     img = load_image_gray(filepath)
#
#     # convert byte to float
#     dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#
#     # use NumPy to rapidly shift DFT diagram and prepare for display FFT diagram
#     dft_shift = np.fft.fftshift(dft)
#     result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
#
#     return [img, result]


def use_notch_filter(dft, x, y, radius):
    # height, weight, channel = dft.shape
    # new_img = np.empty((height, weight))
    thetas = np.linspace(-np.pi, np.pi, 100)

    for r in range(-radius, radius, 1):
        for theta in thetas:
            xpt = r * np.cos(theta) + x
            ypt = r * np.sin(theta) + y

            # round number to integer
            xpt = int(round(xpt))
            ypt = int(round(ypt))

            # assign value
            dft[xpt, ypt] = 0

    return dft


def frequency_noise_analysis(filepath: str):
    # load image from file
    img = load_image_gray(filepath)

    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # use NumPy to rapidly shift DFT diagram and prepare for display FFT diagram
    dft_shift = np.fft.fftshift(dft)

    dft_filter = use_notch_filter(dft_shift.copy(), 40, 55, 5)      # filter noise pt 1
    dft_filter = use_notch_filter(dft_filter, 80, 55, 5)      # filter noise pt 2
    dft_filter = use_notch_filter(dft_filter, 165, 55, 5)     # filter noise pt 3
    dft_filter = use_notch_filter(dft_filter, 205, 55, 5)     # filter noise pt 4

    dft_filter = use_notch_filter(dft_filter, 40, 110, 5)      # filter noise pt 1
    dft_filter = use_notch_filter(dft_filter, 80, 110, 5)      # filter noise pt 2
    dft_filter = use_notch_filter(dft_filter, 165, 110, 5)     # filter noise pt 3
    dft_filter = use_notch_filter(dft_filter, 205, 110, 5)     # filter noise pt 4

    # convert dft image back
    img_back = np.fft.fftshift(dft_filter)
    img_back = cv2.idft(img_back, flags=cv2.DFT_COMPLEX_INPUT | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # prepare image
    dft_img = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    filter_img = 20 * np.log(cv2.magnitude(dft_filter[:, :, 0], dft_filter[:, :, 1]))

    return [img, dft_img, filter_img, img_back]


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "FFT Result")
    plot.append_image(diagrams[2], "Use Notch Filter")
    plot.append_image(diagrams[3], "Final Result")

    plot.show(1, 4)


if __name__ == "__main__":
    _noise_file = "../../Data/DIP/DIP3E_CH04_Original_Images/DIP3E_Original_Images_CH04/Fig0421(" \
                  "car_newsprint_sampled_at_75DPI).tif "

    display_result(frequency_noise_analysis(_noise_file))
