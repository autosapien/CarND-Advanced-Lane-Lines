import cv2
import numpy as np


def abs_sobel_threshold(img, orient='x', sobel_kernel=9, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output


def magnitude_threshold(img, sobel_kernel=9, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 255

    # Return the binary image
    return binary_output


def direction_threshold(img, sobel_kernel=9, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # Return the binary image
    return binary_output


def yellow_mask(img, ret_mask=True):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lower_yellow = np.uint8([18, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    y_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    if ret_mask:
        return y_mask
    return cv2.bitwise_and(img, img, mask=y_mask)


def lab_b_mask(img, thresh=(155, 255), ret_mask=True):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    _, binary = cv2.threshold(b_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    if ret_mask:
        return binary
    return cv2.bitwise_and(img, img, mask=binary)


def color_pipeline_ng(img, s_threshold=(254, 255), sx_threshold=(70, 250)):

    _,_,s_mask = color_pipeline(img, s_threshold=s_threshold, sx_threshold=sx_threshold)
    y_mask = yellow_mask(img)
    labb_mask = lab_b_mask(img)

    # Stack each channel
    res = np.zeros_like(s_mask)
    res[(y_mask == 255) | (labb_mask == 255) | (s_mask == 255)] = 255
    return res


def color_pipeline(img, s_threshold=(254, 255), sx_threshold=(70, 250)):

    # Convert to HLS color spac
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=21)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_threshold[0]) & (scaled_sobel <= sx_threshold[1])] = 255

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 255

    # Stack each channel
    binary = np.zeros_like(s_channel)
    binary[(s_binary == 255) | (sxbinary == 255)] = 255
    return s_binary, sxbinary, binary

