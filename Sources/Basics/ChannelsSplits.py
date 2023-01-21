import cv2
import matplotlib.pyplot as plt


def split_test(img_BGR):
    # convert the BGR sequence to RGB, for plotting
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1); plt.imshow(img_RGB); plt.axis('off'); plt.title('RGB image')

    R = img_RGB[:, :, 0]
    plt.subplot(2, 2, 2); plt.imshow(R, cmap="gray"); plt.axis('off'); plt.title('The red channel')

    G = img_RGB[:, :, 1]
    plt.subplot(2, 2, 3); plt.imshow(G, cmap="gray"); plt.axis('off'); plt.title('The green channel')

    B = img_RGB[:, :, 2]
    plt.subplot(2, 2, 4); plt.imshow(B, cmap="gray"); plt.axis('off'); plt.title('The blue channel')

    plt.show()

