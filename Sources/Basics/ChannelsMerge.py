import cv2
import matplotlib.pyplot as plt


def merge_test(bgr_img, gray_img):

    # convert the gray image's size to the target size
    gray_img = cv2.resize(gray_img, (bgr_img.shape[1], bgr_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # split the bgr channels
    B, G, R = cv2.split(bgr_img)

    # create a new image with alpha channel
    bgr_new = cv2.merge((B, G, R, gray_img))

    # do something more before plotting with plt
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgba_new = cv2.cvtColor(bgr_new, cv2.COLOR_BGRA2RGBA)

    plt.subplot(1, 2, 1); plt.imshow(rgb_img); plt.axis('off'); plt.title('RGB image')
    plt.subplot(1, 2, 2); plt.imshow(rgba_new); plt.axis('off'); plt.title('RGBA image')

    plt.show()