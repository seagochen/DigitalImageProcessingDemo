import cv2
import numpy as np

from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter


def update_dft_with_kernel(dft, kernel):
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


def turbulence_degradation(img, k):
    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # generate turbulence degradation
    width, height, _ = dft.shape  # derive width, height, channel

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

    # apply kernel
    final_dft = update_dft_with_kernel(dft, kernel)

    # convert dft image back
    final_img = cv2.idft(final_dft, flags=cv2.DFT_COMPLEX_INPUT | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # return
    return final_img


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "Violent k=0.0015")
    plot.append_image(diagrams[2], "Recovery from Violent with k=-0.0010")

    plot.show(1, 3)


if __name__ == "__main__":

    # load image from file
    img = load_image_gray(
        "../../Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0525(a)(aerial_view_no_turb).tif")

    img_with_0015 = turbulence_degradation(img, 0.0015)
    img_recovery = turbulence_degradation(img_with_0015, -0.0010)

    display_result((img, img_with_0015, img_recovery))
