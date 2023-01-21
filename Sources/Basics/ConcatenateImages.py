import matplotlib.pyplot as plt
import numpy as np


def concatenation_test(img_gray):
    # load image as a gray scale image
    img_copy = img_gray.copy()

    # concatenate two images by horizontally and vertically
    horizon = np.concatenate((img_gray, img_copy), axis=1)
    vertical = np.concatenate((img_gray, img_copy), axis=0)

    # plot images
    plt.subplot(1, 2, 1);
    plt.imshow(horizon, cmap="gray");
    plt.axis('off');
    plt.title('horizontally')
    plt.subplot(1, 2, 2);
    plt.imshow(vertical, cmap="gray");
    plt.axis('off');
    plt.title('vertically')

    plt.show()
