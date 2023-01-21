import cv2
import matplotlib.pyplot as plt


def cropping_test(origin):
    # convert the BGR to RGB
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

    # crop a piece from origin image
    cropped = origin[0:128, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]

    # show the images
    plt.subplot(1, 2, 1);
    plt.imshow(origin);
    plt.axis('off');
    plt.title('origin')
    plt.subplot(1, 2, 2);
    plt.imshow(cropped);
    plt.axis('off');
    plt.title('cropped')
    plt.show()
