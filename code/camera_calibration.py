import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import undistort_image, camera_calibration_save

# Number of corners in the given images
nx = 9
ny = 6

# Calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Step through the calibration images and search for chessboard corners to get objpoints and imgpoints
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Test undistortion on an image
img = cv2.imread('../camera_cal/calibration1.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
undistorted = undistort_image(img, mtx, dist)

# Draw visualization
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(undistorted)
ax2.set_title('Undistored Image', fontsize=10)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
f.savefig('../output_images/calibrated.jpg')

# Save the camera calibration result for later use
file = "../camera_cal/calibration.p"
camera_calibration_save(file, mtx, dist)
print("Calibration Matrix and Distortion Coefficient save to: ", file)
