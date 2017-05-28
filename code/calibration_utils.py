import cv2
import pickle


def undistort_image(img, mtx, dist):
    """ Undistort an image given its Calibration Matrix and Distortion Coefficient
    """
    img_size = (img.shape[1], img.shape[0])
    return cv2.undistort(img, mtx, dist, None, mtx)


def camera_calibration_save(file, mtx, dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open(file, "wb" ) )


def camera_calibration_load(file):
    dist_pickle = pickle.load(open(file, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist


def load_and_undistort_img(img_file, mtx, dist):
    """Read a file from disk and undistort
    """
    return cv2.undistort(cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB), mtx, dist, None, mtx)


def undistort_img(img, mtx, dist):
    """Undistort
    """
    return cv2.undistort(img, mtx, dist, None, mtx)
