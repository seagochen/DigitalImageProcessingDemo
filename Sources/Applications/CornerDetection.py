import cv2
import numpy as np

img = cv2.imread("../../Data/DIP/DIP3E_CH02_Original_Images/DIP3E_Original_Images_CH02/Fig0219(rose1024).tif",
                 cv2.IMREAD_GRAYSCALE)

# test image exist
if img is None:
    print("image cannot be load")
    exit(0)

# get the image information
print(f"image matrix with rows: {img.shape[0]} and cols:{img.shape[1]}")

gray = np.float32(img)

feature_params = dict(maxCorners=100,  # Maximum number of corners to return. If there are more corners than are found,
                      # the strongest of them is returned.

                      qualityLevel=0.01,  # Parameter characterizing the minimal accepted quality of image corners.
                      # The parameter value is multiplied by the best corner quality measure, which is the minimal
                      # eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ).
                      # The corners with the quality measure less than the product are rejected.
                      # For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 ,
                      # then all the corners with the quality measure less than 15 are rejected.

                      minDistance=10,  # Minimum possible Euclidean distance between the returned corners.

                      blockSize=15  # Size of an average block for computing a derivative covariation matrix
                      # over each pixel neighborhood.
                      )

corners = cv2.goodFeaturesToTrack(gray, **feature_params)

# color output
color_output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(color_output, (x, y), 3, (0, 0, 255))

cv2.imshow('Corner Detection', color_output)
cv2.waitKey(0)
