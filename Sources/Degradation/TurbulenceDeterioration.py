import cv2
import numpy as np

from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter


def show_dft(original_dft, updated_dft):
    # use NumPy to rapidly shift DFT diagram and prepare for display FFT diagram
    dft_shift1 = np.fft.fftshift(original_dft)
    dft_shift2 = np.fft.fftshift(updated_dft)

    # prepare image
    dft_shift1 = 20 * np.log(cv2.magnitude(dft_shift1[:, :, 0], dft_shift1[:, :, 1]))
    dft_shift2 = 20 * np.log(cv2.magnitude(dft_shift2[:, :, 0], dft_shift2[:, :, 1]))

    # display result
    plot = DiagramPlotter()
    plot.append_image(dft_shift1, "Original")
    plot.append_image(dft_shift2, "Updated")
    plot.show(1, 2)


def degradation_kernel(dft, k):
    # derive width, height, channel
    width, height, _ = dft.shape

    # center pointer
    p = width / 2 + 1.0
    q = height / 2 + 1.0

    # generate an empty kernel
    kernel = np.zeros((width, height), dtype=np.float32)

    # generate turbulence kernel
    for u in range(width):
        for v in range(height):
            power = -k * np.power((u - p) ** 2 + (v - q) ** 2, 5 / 6)
            kernel[u, v] = np.power(np.e, power)

    return kernel


def update_dft_with_degradation(dft, kernel):

    # derive width, height, channel
    width, height, _ = dft.shape

    # shift dft
    dft_backup = np.fft.fftshift(dft)

    # apply the kernel
    dft_backup[:, :, 0] = dft_backup[:, :, 0] * kernel
    dft_backup[:, :, 1] = dft_backup[:, :, 1] * kernel

    # shift back
    dft_backup = np.fft.fftshift(dft_backup)

    return dft_backup


def turbulence_deterioration(img, k):

    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # generate turbulence degradation
    kernel = degradation_kernel(dft, k)

    # apply kernel
    final_dft = update_dft_with_degradation(dft, kernel)

    # convert dft image back
    final_img = cv2.idft(final_dft, flags=cv2.DFT_COMPLEX_INPUT | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # return
    return final_img


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "Violent k=0.0025")
    plot.append_image(diagrams[2], "Medium k=0.001")
    plot.append_image(diagrams[3], "Slight k=0.00025")

    plot.show(2, 2)


if __name__ == "__main__":

    # load image from file
    img = load_image_gray(
        "../../Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0525(a)(aerial_view_no_turb).tif")

    img_with_0025 = turbulence_deterioration(img, 0.0025)
    img_with_001 = turbulence_deterioration(img, 0.001)
    img_with_00025 = turbulence_deterioration(img, 0.00025)

    display_result((img, img_with_0025, img_with_001, img_with_00025))
