import cv2
import numpy as np

from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter

from Cv.Restoration.TurbulenceRecover import turbulence_degradation, update_dft_with_kernel


def img_to_dft(img):
    # convert byte to float
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    return dft


def dft_to_img(dft):
    # convert dft image back
    return cv2.idft(dft, flags=cv2.DFT_COMPLEX_INPUT | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


def create_parameters(dft):

    # generate empty matrix
    weight = np.ones_like(dft, dtype=np.float32)

    # return to caller
    return weight


def compute_loss(dft_predicated, dft_target):
    # compute mse of original and recovery images
    new_errors = (dft_predicated - dft_target)**2

    # return sum
    return np.sum(new_errors)


def backward(weights, dft_predicated, dft_target, lambs):

    # update value chain
    updated_val = np.absolute(lambs * (dft_predicated - dft_target))

    width, height, channel = dft_predicated.shape
    for w in range(width):
        for h in range(height):
            for c in range(channel):

                if dft_target[w, h, c] >= dft_predicated[w, h, c] >= 0:

                    if weights[w, h, c] >= 0:
                        weights[w, h, c] = weights[w, h, c] + updated_val[w, h, c]
                    else:
                        weights[w, h, c] = weights[w, h, c] - updated_val[w, h, c]

                    continue

                elif dft_predicated[w, h, c] >= dft_target[w, h, c] >= 0:

                    if weights[w, h, c] >= 0:
                        weights[w, h, c] = weights[w, h, c] - updated_val[w, h, c]
                    else:
                        weights[w, h, c] = weights[w, h, c] + updated_val[w, h, c]

                    continue

                elif 0 >= dft_target[w, h, c] >= dft_predicated[w, h, c]:

                    if weights[w, h, c] >= 0:
                        weights[w, h, c] = weights[w, h, c] - updated_val[w, h, c]
                    else:
                        weights[w, h, c] = weights[w, h, c] + updated_val[w, h, c]

                    continue

                elif 0 >= dft_predicated[w, h, c] >= dft_target[w, h, c]:

                    if weights[w, h, c] >= 0:
                        weights[w, h, c] = weights[w, h, c] + updated_val[w, h, c]
                    else:
                        weights[w, h, c] = weights[w, h, c] - updated_val[w, h, c]

                    continue

                else:
                    # dft_target[w, h, c] > 0 >= dft_predicated[w, h, c]:
                    # dft_predicated[w, h, c] > 0 >= dft_target[w, h, c]:

                    if weights[w, h, c] >= 0:
                        weights[w, h, c] = weights[w, h, c] - updated_val[w, h, c]
                    else:
                        weights[w, h, c] = weights[w, h, c] + updated_val[w, h, c]

                    continue

    # return the updated weights
    return weights


def forward(weights, dft_origin):
    return weights * dft_origin  # predicated_y = weight * 1


def normalize_data(dft1, dft2):
    max_dft = np.maximum(dft1, dft2)
    max_data = np.max(max_dft.reshape(-1))

    min_dft = np.minimum(dft1, dft2)
    min_data = np.min(min_dft.reshape(-1))

    length = max_data - min_data

    dft1 = dft1 / length * 100.0
    dft2 = dft2 / length * 100.0

    return dft1, dft2, length


def gradient_descent(img, distort):

    dft_origin, dft_distort, length = normalize_data(img_to_dft(img), img_to_dft(distort))

    # lambda rate
    lambs = 0.01

    # create weight matrices for real and image parts of DFT and set update rate to 0.015
    weight = create_parameters(dft_origin)

    # loop and descent gradients for both real and image weights
    for i in range(500):

        dft_predicated = forward(weight, dft_origin)

        # compute loss
        mse = compute_loss(dft_predicated, dft_distort)

        # backward computation
        weight = backward(weight, dft_predicated, dft_distort, lambs)

        # print out debug message
        print("mse: %.4f" % mse)
        # print("\n")

    return weight * length


def display_result(diagrams):
    plot = DiagramPlotter()

    plot.append_image(diagrams[0], "Original")
    plot.append_image(diagrams[1], "Violent k=0.0015")
    plot.append_image(diagrams[2], "Recovery from Violent with k=-0.0010")
    plot.append_image(diagrams[2], "Recovery from LMS Filter")

    plot.show(1, 4)


def recovery_image(img, distorted, disrecv):
    # copy backup
    distorted_backup = distorted.copy()

    # obtain weight
    weight = gradient_descent(img, distorted_backup)

    # use LMS filter to recovery distorted image
    dft = img_to_dft(distorted)
    dft = dft / weight
    img_ds = dft_to_img(dft)

    # show result
    display_result((img, distorted, disrecv, img_ds))


if __name__ == "__main__":

    # load image from file
    img = load_image_gray(
        "../../Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/Fig0525(a)(aerial_view_no_turb).tif")

    img_with_0015 = turbulence_degradation(img, 0.0015)
    img_recovery = turbulence_degradation(img_with_0015, -0.0010)

    # show recovery
    recovery_image(img, img_with_0015, img_recovery)
