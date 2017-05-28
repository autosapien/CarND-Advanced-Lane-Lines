import glob
import cv2
import matplotlib.pyplot as plt
from calibration_utils import camera_calibration_load, load_and_undistort_img

input_files = glob.glob('../test_images/*.jpg')
output_dir = '../output_images/'

# Load calibration details, load images from input dir and undistort them
mtx, dist = camera_calibration_load('../camera_cal/calibration.p')
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in input_files]
undistorted = [load_and_undistort_img(f, mtx, dist) for f in input_files]

# Save visualization
f, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 8 ))
f.tight_layout()
ax1[0].imshow(imgs[6])
ax1[0].set_title('Original Image', fontsize=14)
ax1[1].imshow(undistorted[6])
ax1[1].set_title('Undistored Image', fontsize=14)
ax2[0].imshow(imgs[7])
ax2[0].set_title('Original Image', fontsize=14)
ax2[1].imshow(undistorted[7])
ax2[1].set_title('Undistored Image', fontsize=14)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.)
f.savefig('../output_images/road_undistorted.jpg')
