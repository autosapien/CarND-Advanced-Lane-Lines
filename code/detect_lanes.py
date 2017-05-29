import glob
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from calibration_utils import camera_calibration_load, load_and_undistort_img, undistort_img
from image_color_transforms import color_pipeline, color_pipeline_ng, yellow_mask, lab_b_mask
from perspective_utils import image_add_offset, bird_view
from lane_marking_utils import fit_poly, plot_lanes_on_road_image, curvature_lr, distance_from_center
import matplotlib.pyplot as plt

input_files = glob.glob('../test_images/*.jpg')
output_dir = '../output_images/'
viz = True


def annotate_road_image(image):
    undist = undistort_img(image, mtx, dist)
    warped = bird_view(undist, M)
    warped_binary = color_pipeline_ng(warped)
    left_fit, right_fit = fit_poly(np.uint8(warped_binary / 255))
    left_cur, right_cur = curvature_lr(warped_binary, left_fit, right_fit)
    dfc, dir = distance_from_center(warped.shape[0], warped.shape[1], left_fit, right_fit)
    result = plot_lanes_on_road_image(warped_binary, undist=undist, Minv=Minv,
                                      left_fit=left_fit, right_fit=right_fit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Radius of Left Curve: {0:.2f} m".format(left_cur), (30, 50), font, 1, (245, 255, 245), 2)
    cv2.putText(result, "Radius of Right Curve: {0:.2f} m".format(right_cur), (30, 90), font, 1, (245, 255, 245), 2)
    cv2.putText(result, "Vehicle is {0:.2f} m {1} of center".format(dfc, dir), (30, 130), font, 1, (245, 255, 245), 2)
    return result


# Load calibration details, load images from input dir and undistort them
mtx, dist = camera_calibration_load('../camera_cal/calibration.p')
imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in input_files]
undistorted = [load_and_undistort_img(f, mtx, dist) for f in input_files]

# Undistort an image form the road
if viz:
    road = load_and_undistort_img('../test_images/test4.jpg', mtx, dist)
    plt.imsave('../output_images/road_undistorted.jpg', road)

# first do some arithmetic on the vanishing point to get the trapezoid points
vp_y = 418  # vp from top of image
indent_x = -30  # trapezoid base indent into image (negative value go outside)
delta = 34  # distance to trapezoid top from vp
t = vp_y + delta  # top pf trapezoid from top of image
s = ((640-indent_x)*delta) / (720-vp_y)  # length of trapezoid top / 2
src = np.float32([[640-s, t], [640+s, t], [1280-indent_x, 720], [indent_x, 720]])

# The trapezoid is transformed into a rectangle in the bird view, compute the coordinates of the vertices of this rect
bird_view_size = (800, 800)  # Setup a 800x600 image to look at the birds view
offsets = [40, 0, 40, 0]  # where the trapezoid vertices should be transformed to from the outside of the image
dst = np.float32(image_add_offset(bird_view_size, offsets))

# Draw trapezoid on an image with straight lanes
if viz:
    img = np.copy(undistorted[0])
    pts = np.array(src, np.int32)
    cv2.polylines(img, [pts], True, (255, 0, 255), thickness=1)
    f = plt.figure(figsize=[4,3])
    plt.imsave('../output_images/trapezoid.jpg', img)

# Get transform matrix and invese transform matrix
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Do perspective transform on image with marker
if viz:
    warped_with_marker = bird_view(img, M, dsize=bird_view_size)
    plt.imsave('../output_images/birds_view_with_trapezoid.jpg', warped_with_marker)


# Show what transformation does to other images
if viz:
    imgs_to_examine = [3,5,6,7]
    f, axes = plt.subplots(len(imgs_to_examine), 2, figsize=(25, 32))
    f.tight_layout()
    for i, index in enumerate(imgs_to_examine):
        img = undistorted[index]
        axes[i][0].set_title('Original Image', fontsize=19)
        axes[i][0].imshow(img)
        axes[i][0].axis('off')
        axes[i][1].set_title('Birds View Image', fontsize=19)
        axes[i][1].imshow(bird_view(img, M))
        axes[i][1].axis('off')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.subplots_adjust(hspace=.1, wspace=.05)
    f.savefig('../output_images/birds_view.jpg')


# Now run the color pipeline on these images to see what we get:
if viz:
    f, axes = plt.subplots(len(imgs_to_examine), 5, figsize=(55, 45))
    f.tight_layout()
    fs = 45
    for i, index in enumerate(imgs_to_examine):
        img = bird_view(undistorted[index], M)
        yellow = yellow_mask(img)
        lab_b = lab_b_mask(img)
        _,_,sobel = color_pipeline(img)
        stacked = color_pipeline_ng(img)
        axes[i][0].set_title('Birds View Image', fontsize=fs)
        axes[i][0].imshow(img)
        axes[i][0].axis('off')
        axes[i][1].set_title('Yellow Mask', fontsize=fs)
        axes[i][1].imshow(yellow)
        axes[i][1].axis('off')
        axes[i][2].set_title('Lab B Channel Mask', fontsize=fs)
        axes[i][2].imshow(lab_b)
        axes[i][2].axis('off')
        axes[i][3].set_title('Sobel X Operator (k=21)', fontsize=fs)
        axes[i][3].imshow(sobel)
        axes[i][3].axis('off')
        axes[i][4].set_title('Stacked Together', fontsize=fs)
        axes[i][4].imshow(stacked)
        axes[i][4].axis('off')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.subplots_adjust(hspace=.1, wspace=.05)
    f.savefig('../output_images/birds_view_masked.jpg')

if viz:
    histogram = np.sum(stacked[stacked.shape[0]//2:,:], axis=0)
    f = plt.figure(figsize=(4,3))
    plt.plot(histogram/255)  # only for display purposes show each pixel as 1 instead of 255
    plt.savefig('../output_images/histogram.jpg')

if viz:
    imgs_to_examine = [3,5,6,7]
    for i, index in enumerate(imgs_to_examine):
        warped = bird_view(undistorted[index], M)
        warped_binary = color_pipeline_ng(warped)
        filename = "../output_images/lanes_marked_{}.jpg".format(index)
        img = fit_poly(warped_binary, visualize=True, filename=filename)

if viz:
    annotated = annotate_road_image(undistorted[7])
    plt.imsave('../output_images/annotated7.jpg', annotated)
    annotated = annotate_road_image(undistorted[0])
    plt.imsave('../output_images/annotated0.jpg', annotated)


if not viz:
    in_vid = '../harder_challenge_video.mp4'
    out_vid = '../processed_' + in_vid.split('/')[-1]
    clip = VideoFileClip(in_vid)
    annotated_clip = clip.fl_image(annotate_road_image)
    annotated_clip.write_videofile(out_vid, audio=False)
