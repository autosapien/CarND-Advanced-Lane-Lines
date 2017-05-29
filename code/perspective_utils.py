import cv2


def image_add_offset(size, offset):
    """Add offsets to the 4 vertices of an image. The ordering of the vertices and offsets is rt, lt, rb, lb
    """
    lt = [offset[0], offset[1]]
    rt = [size[0] - offset[2], offset[1]]
    rb = [size[0] - offset[2], size[1] - offset[3]]
    lb = [offset[0], size[1] - offset[3]]
    return [lt, rt, rb, lb]


def bird_view(img, transformation_matrix, dsize=(800, 800)):
    """Return a binary image after warping the image with applying the transformation matrix
    """
    return cv2.warpPerspective(img, transformation_matrix, dsize=dsize)
