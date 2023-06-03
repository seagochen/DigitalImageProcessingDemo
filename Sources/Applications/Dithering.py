import cv2
import numpy as np


#################### Floyd-Steinberg Dithering ####################

def floyd_steinberg_dithering_kernel(image):
    for y in range(image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            old_pixel = image[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            image[y, x] = new_pixel
            error = old_pixel - new_pixel
            image[y, x + 1] += error * 7 / 16
            image[y + 1, x - 1] += error * 3 / 16
            image[y + 1, x] += error * 5 / 16
            image[y + 1, x + 1] += error * 1 / 16
    return image


def floyd_steinberg_dithering():
    # Load an RGB image
    image = cv2.imread("Data/test.png")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Floyd-Steinberg dithering
    dithered_image = floyd_steinberg_dithering_kernel(np.copy(gray_image))

    # Display the original grayscale image and the dithered image
    cv2.imshow("Original", gray_image)
    cv2.imshow("FS Dithered", dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#################### Atkinson Dithering ####################


def atkinson_dithering_kernel(image):
    error = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0] - 2):
        for x in range(image.shape[1] - 2):
            old_pixel = image[y, x] + error[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            image[y, x] = new_pixel
            diff = old_pixel - new_pixel
            error[y, x + 1] += diff * 1 / 8
            error[y, x + 2] += diff * 1 / 8
            error[y + 1, x - 1] += diff * 1 / 8
            error[y + 1, x] += diff * 1 / 8
            error[y + 1, x + 1] += diff * 1 / 8
            error[y + 2, x] += diff * 1 / 8
    return image


def atkinson_dithering():
    # Load an RGB image
    image = cv2.imread("Data/test.png")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Atkinson dithering
    dithered_image = atkinson_dithering_kernel(np.copy(gray_image.astype(np.float32)))

    # Display the original grayscale image and the dithered image
    cv2.imshow("Original", gray_image)
    cv2.imshow("A Dithered", dithered_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load an RGB image
    image = cv2.imread("Data/Others/test.png")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Floyd-Steinberg dithering
    fs_dithered_image = floyd_steinberg_dithering_kernel(np.copy(gray_image))

    # Apply Atkinson dithering
    atk_dithered_image = atkinson_dithering_kernel(np.copy(gray_image.astype(np.float32)))

    # Display the original grayscale image and the dithered image
    cv2.imshow("Original", gray_image)
    cv2.imshow("FS Dithered", fs_dithered_image)
    cv2.imshow("ATK Dithered", atk_dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # floyd_steinberg_dithering()
    # atkinson_dithering()

    main()
