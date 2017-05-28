## Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in `code/camera_calibration.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![camera calibration][output_images/calibrated.jpg]

The calibration needs to be done only once. The results are save in `camera_ca/calibration.p` for further use

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, an image from the `test_images\` directory is loaded and distortion correction is applied on it in `test_calibration_on_road.py`.
The calibration details loads the Calibration Matrix and Distortion Coefficient for this camera from  `camera_ca/calibration.p` and undistorts the image by applying the `cv2.undistort()` function
Here we see that the results are not as clearly appreciated as on a chess board image. Look closely at the white car on the right and the dashboard of the driving car one can see distortion correction.
 
![distortion correction on road][output_images/road_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The goal is to identify lanes on the road. We use a combination of tow approaches. Image color tranforms and Sobel operator transformation to find near vertical edges (lanes are vertical)
In order to select the most useful transforms I tested various transforms out in `transform_selection_for_lane_detection.py`
 
 
From http://vanseodesign.com/web-design/hue-saturation-and-lightness/ we learn that 
> Saturation refers to how pure or intense a given hue is. 100% saturation means there’s no addition of gray to the hue. The color is completely pure. At the other extreme a hue with 0% saturation appears as a medium gray. The more saturated (closer to 100%) a color is, the more vivid or brighter it appears. Desaturated colors, on the other hand, appear duller

We can play with saturation, lightness and hue on this link https://www.w3schools.com/colors/colors_hsl.asp

Anything that needs to stand out on a dark surface or needs to be visible in the dark needs to have a high saturation and medium to high lightness. Road lanes clearly fit into that category.

Again for anything to stand out on a dark surface needs to have some amount of brightness. The V channels in HSV colorspace specifies the brightness of the image.

In addition, we use a Sobel operator in the X direction, this isolates edges that are present in the y (vertical) direction in the image. Lanes lines are generally vertical.

We will use a combination of these to find lanes on the road. Let use see a few examples of these transforms on some of the simple and troublesome images 

![color space][output_images/color_transform_test.jpg]

As anticipated the saturation in HLS looks to be the most promising transform to find lanes. 

![soble x][output_images/sobel_transform_test.jpg]

And the Sobel X seems to work well to identify horizontal lines

We now stack these images onto one another to yield a more complete picture 

![stacked][output_images/stacked.jpg]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Now we move to transform the view of the dash cam to a birds eye view. 
This allows for better identification and fitting of the lines.

The key to getting a a good transform is the right identification of the vanishing point (where the train tracks would join) in the source image.
In `test_images\straight_lines1.jpg` we identify that the vanishing point is at 420 px from the top.

Based on this we can use any isosceles trapezoid with its base centered at the bottom of the image and its sized converging to the vanishing point.

We use one which covers almsst all of the cameras wide angle view as seen below:

![stacked][output_images/trapezoid.jpg]
![bird view with marker][output_images/birds_view_with_trapezoid.jpg]

Taking a wide view has an added advantage we can use this transform to restrict the region of interest in one pass.
Here we have accepted data from 60 pixels outside the trapezoid on the sides and 20 px from top. nothing from below as the trapezoid is at the bottom of the image. 
This can be seen in `code/detect.lanes.py` 
```
bird_view_size = (600, 600)  # Setup a 600x600 image to look at the birds view
offsets = [60, 20, -60, 0]
```

Transforming our set of trouble images into birds eye view would result in
 
![bird view][output_images/birds_view.jpg]

We can see that the lanes are parallel and pretty clear at the bottom of the views (near the camera of car), looks good so far.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now we can fit a second degree polynomial 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
