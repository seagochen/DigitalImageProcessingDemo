import cv2
import numpy as np
import matplotlib.pyplot as plt

def skin_detection(image):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin colors
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin, skin_mask

def main():
    # Load an RGB image
    image = cv2.imread("Data/Others/people.png")

    # Perform skin detection
    skin, skin_mask = skin_detection(image)

    # Create a new figure with a specific size (in inches)
    plt.figure(figsize=(10, 15))

    # Display the original image
    plt.subplot(311), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    # Display the skin mask
    plt.subplot(312), plt.imshow(skin_mask, cmap="gray"), plt.title("Mask")
    plt.xticks([]), plt.yticks([])

    # Display the image with skin detection
    plt.subplot(313), plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)), plt.title("Skin Detection")
    plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()
