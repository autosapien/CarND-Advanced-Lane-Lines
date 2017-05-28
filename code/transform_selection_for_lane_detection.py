import glob
import cv2
import matplotlib.pyplot as plt
from calibration_utils import camera_calibration_load, load_and_undistort_img
from image_color_transforms import abs_sobel_threshold, magnitude_threshold, direction_threshold, pipeline

input_files = glob.glob('../test_images/*.jpg')
output_dir = '../output_images/'


mtx, dist = camera_calibration_load('../camera_cal/calibration.p')
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in input_files]
undistorted = [load_and_undistort_img(f, mtx, dist) for f in input_files]

imgs_to_examine = [3,5,6,7]


# Look at color spaces
f, axes = plt.subplots(len(imgs_to_examine), 4, figsize=(30, 20))
f.tight_layout()
for i, index in enumerate(imgs_to_examine):
    axes[i][0].set_title('Original Image', fontsize=19)
    axes[i][0].imshow(undistorted[index])
    axes[i][0].axis('off')
    axes[i][1].set_title('HLS - Saturation Channel', fontsize=19)
    axes[i][1].imshow(cv2.cvtColor(undistorted[index], cv2.COLOR_RGB2HLS)[:,:,2], "gray")
    axes[i][1].axis('off')
    axes[i][2].set_title('HLS - Lightness Channel', fontsize=19)
    axes[i][2].imshow(cv2.cvtColor(undistorted[index], cv2.COLOR_RGB2HLS)[:,:,1], "gray")
    axes[i][2].axis('off')
    axes[i][3].set_title('HSV - Value Channel', fontsize=19)
    axes[i][3].imshow(cv2.cvtColor(undistorted[index], cv2.COLOR_RGB2HSV)[:,:,2], "gray")
    axes[i][3].axis('off')
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
plt.subplots_adjust(hspace=.1, wspace=.05)
f.savefig('../output_images/color_transform_test.jpg')


# Look at Sobel transfroms
f, axes = plt.subplots(len(imgs_to_examine), 4, figsize=(30, 20))
f.tight_layout()
for i, index in enumerate(imgs_to_examine):
    axes[i][0].set_title('Original Image', fontsize=19)
    axes[i][0].imshow(undistorted[index])
    axes[i][0].axis('off')
    axes[i][1].set_title('Sobel (k=9) along X axis. threshold (20,255)', fontsize=19)
    axes[i][1].imshow(abs_sobel_threshold(cv2.cvtColor(undistorted[index], cv2.COLOR_BGR2RGB), thresh=(20,255)),  "gray")
    axes[i][1].axis('off')
    axes[i][2].set_title('Sobel (k=9) along Y axis. Threshold (40,255)', fontsize=19)
    axes[i][2].imshow(abs_sobel_threshold(cv2.cvtColor(undistorted[index], cv2.COLOR_BGR2RGB), 'y', thresh=(40,255)),  "gray")
    axes[i][2].axis('off')
    axes[i][3].set_title('Sobel (k=9) magnitude. Threshold (40,255)', fontsize=19)
    axes[i][3].imshow(magnitude_threshold(cv2.cvtColor(undistorted[index], cv2.COLOR_BGR2RGB), thresh=(40, 255)),  "gray")
    axes[i][3].axis('off')
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
plt.subplots_adjust(hspace=.1, wspace=.05)
f.savefig('../output_images/sobel_transform_test.jpg')


# Stack sobel X and HLS saturation
f, axes = plt.subplots(len(imgs_to_examine), 4, figsize=(30, 20))
f.tight_layout()
for i, index in enumerate(imgs_to_examine):
    s_binary, sx_binary, binary = pipeline(cv2.cvtColor(undistorted[index], cv2.COLOR_BGR2RGB), s_threshold=(170, 255),
                                           sx_threshold=(20, 255))
    axes[i][0].set_title('Original Image', fontsize=19)
    axes[i][0].imshow(undistorted[index])
    axes[i][0].axis('off')
    axes[i][1].set_title('HLS - Saturation Channel. threshold (170,255)', fontsize=19)
    axes[i][1].imshow(s_binary, "gray")
    axes[i][1].axis('off')
    axes[i][2].set_title('Sobel (k=9) along X axis. Threshold (20,255)', fontsize=19)
    axes[i][2].imshow(sx_binary, "gray")
    axes[i][2].axis('off')
    axes[i][3].set_title('Stacked Sobel X & HLS Saturation', fontsize=19)
    axes[i][3].imshow(binary, "gray")
    axes[i][3].axis('off')
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
plt.subplots_adjust(hspace=.1, wspace=.05)
f.savefig('../output_images/stacked.jpg')