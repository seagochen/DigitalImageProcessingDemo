import cv2
import numpy as np

# 读取测试图像
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ConstrainedLeastSquaresFiltering(img):
    # Add Gaussian noise to the image
    noisy_img = img + np.random.normal(0, 20, size=img.shape)

    # Define the degradation matrix
    degradation_matrix = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.1, 0.9]])

    # Define the constraint matrix
    constraint_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Define the regularization parameter
    alpha = 0.1

    # Compute the inverse of the degradation matrix
    degradation_matrix_inv = np.linalg.inv(degradation_matrix)

    # Compute the constrained least squares estimate
    cls_estimate = cv2.filter2D(noisy_img, -1, degradation_matrix_inv)
    cls_estimate = cls_estimate - alpha * cv2.filter2D(cls_estimate, -1, constraint_matrix)

    # Convert the image to uint8 and clip it to [0, 255]
    cls_estimate = np.clip(cls_estimate, 0, 255).astype(np.uint8)

    # Show the original image, the noisy image, and the restored image
    cv2.imshow("Original Image", img)
    cv2.imshow("Noisy Image", noisy_img)
    cv2.imshow("Restored Image", cls_estimate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function
ConstrainedLeastSquaresFiltering(img)
