import glob
import cv2
import numpy as np
from calibration_utils import camera_calibration_load, load_and_undistort_img, undistort_img
from image_color_transforms import color_pipeline
from perspective_utils import image_add_offset, bird_view

import matplotlib.pyplot as plt

input_files = glob.glob('../test_images/*.jpg')
output_dir = '../output_images/'

def pipeline(img, mtx, dist, trans_mtx, ):
    img = undistort_img(img)


# Load calibration details, load images from input dir and undistort them
mtx, dist = camera_calibration_load('../camera_cal/calibration.p')
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in input_files]
undistorted = [load_and_undistort_img(f, mtx, dist) for f in input_files]

# first do some arithmetic on the vanishing point to get the trapezoid points
vp_y = 420
indent_x = 10
delta = 20
t = vp_y + delta
s = ((640-indent_x)*delta) / (720-vp_y)
src = np.float32([[640-s, t], [640+s, t], [1280-indent_x, 720], [indent_x, 720]])

# The trapezoid is transformed into a rectangle in the bird view, compute the coordinates of the vertices of this rect
bird_view_size = (600, 600)  # Setup a 600x600 image to look at the birds view
offsets = [60, 20, -60, 0]


dst = np.float32(image_add_offset(bird_view_size, offsets))

# Draw trapezoid on an image with straight lanes
img = np.copy(undistorted[0])
pts = np.array(src, np.int32)
cv2.polylines(img, [pts], True, (255, 0, 255), thickness=1)
f = plt.figure(figsize=[4,3])
plt.imsave('../output_images/trapezoid.jpg', img)

# Get transform matrix and do perspective transform on image with marker
M = cv2.getPerspectiveTransform(src, dst)
warped_with_marker = bird_view(img, M)
plt.imsave('../output_images/birds_view_with_trapezoid.jpg', warped_with_marker)


# Show what transformation does to other images
imgs_to_examine = [3,5,6,7]
f, axes = plt.subplots(len(imgs_to_examine), 3, figsize=(22, 30))
f.tight_layout()
for i, index in enumerate(imgs_to_examine):
    img = undistorted[index]
    s_binary, sx_binary, binary = color_pipeline(img, s_threshold=(170, 255), sx_threshold=(20, 255))
    axes[i][0].set_title('Original Image', fontsize=19)
    axes[i][0].imshow(img)
    axes[i][0].axis('off')
    axes[i][1].set_title('Birds View Image', fontsize=19)
    axes[i][1].imshow(bird_view(img, M), "gray")
    axes[i][1].axis('off')
    axes[i][2].set_title('Birds View on Color Transformed Image', fontsize=19)
    axes[i][2].imshow(bird_view(binary, M), "gray")
    axes[i][2].axis('off')
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
plt.subplots_adjust(hspace=.1, wspace=.05)
f.savefig('../output_images/birds_view.jpg')


#top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)

