import cv2
import numpy as np

from Utilities.DiagramPlotter import DiagramPlotter


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "vertical")
    plot.append_image(diagrams[2], "horizontal")

    plot.show(1, 3)


if __name__ == "__main__":

    # # load image from file
    # img = load_image_gray(
    #     "../../Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0526(a)(original_DIP).tif")
    # img_with_ab0d1_t1 = motion_deterioration(img, 0.1, 0.1, 1)
    # display_result((img, img_with_ab0d1_t1))

    # specify the kernel size
    kernel_size = 30

    # create the vertical kernel.
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_v = kernel_h.copy()

    # fill the middle row with ones
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)

    # normalize
    kernel_h /= kernel_size
    kernel_v /= kernel_size

    # load image
    img = cv2.imread("../../Data/jpegs/car.jpg")

    # apply the vertical kernel.
    horizontal = cv2.filter2D(img, -1, kernel_h)
    vertical = cv2.filter2D(img, -1, kernel_v)

    # change bgr channels to rgb
    horizontal = cv2.cvtColor(horizontal, cv2.COLOR_BGR2RGB)
    vertical = cv2.cvtColor(vertical, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plot results
    display_result((img, vertical, horizontal))
