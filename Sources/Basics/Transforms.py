import cv2
import matplotlib.pyplot as plt


def rotate(image, angle, center=None, scale=1.0):
    # get the image size
    h, w, _ = image.shape

    # default: set the w/2 and h/2 as center
    if center is None:
        center = (w / 2, h / 2)

    # calculate the rotation matrix
    rotated_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # apply with the rotation matrix
    rotated_image = cv2.warpAffine(image, rotated_matrix, (w, h))

    # return to caller
    return rotated_image


def show_image(image, title, pos):
    plt.subplot(1, 4, pos)
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)


def show_image2(image, title, pos):
    plt.subplot(2, 2, pos)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title(title)


def reshape_image(image):
    show_image2(image, "Original", 1)

    # shrink to its half size
    shrunk_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    show_image2(shrunk_image, "1/4 size", 2)

    # amplify the image with linear interpolation method
    amplified_1 = cv2.resize(shrunk_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    show_image2(amplified_1, "Amplified with linear", 3)

    # amplify the image with cubic interpolation method
    amplified_2 = cv2.resize(shrunk_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    show_image2(amplified_2, "Amplified with cubic", 4)


def show_rotated_images(logo):
    # convert to RGB first
    logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

    # 将原图旋转不同角度
    show_image(logo, "Original", 1)
    show_image(rotate(logo, 45), "45˚", 2)
    show_image(rotate(logo, 90), "90˚", 3)
    show_image(rotate(logo, 125), "125˚", 4)

    # show the results
    plt.show()


def show_reshaped_images(logo):
    # reshape the images
    reshape_image(logo)

    # plot results
    plt.show()
