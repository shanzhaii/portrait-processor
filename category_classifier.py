import cv2
import mediapipe as mp
import math
import glob
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

# documentation https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter/python

def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


# Performs resizing and showing the image
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('image', img)


if __name__ == "__main__":
    # path

    # Height and width that will be used by the model
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    ext = ['png', 'jpg']
    path = 'input/'
    filenames = []
    [filenames.extend(glob.glob(path + '*.' + e)) for e in ext]

    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

    # Create the options that will be used for InteractiveSegmenter
    base_options = python.BaseOptions(model_asset_path='models/magic_touch.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=True)

    x = 0.5
    y = 0.5

    # Create the interactive segmenter
    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        for filename in filenames[:1]:
            # print(img[7:-4])
            # Create the MediaPipe image file that will be segmented
            image = mp.Image.create_from_file(filename)

            # Retrieve the masks for the segmented image
            roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                                   keypoint=NormalizedKeypoint(x, y))
            segmentation_result = segmenter.segment(image, roi)
            category_mask = segmentation_result.category_mask
            mask = 255 - np.stack((category_mask.numpy_view(),), axis=-1)
            cv_image = cv2.imread(filename)
            b_channel, g_channel, r_channel = cv2.split(cv_image)
            img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask))
            cv2.imwrite("output/{image_name}.png".format(image_name=filename[6:-4]), img_BGRA)

            # # Draw a white dot with black border to denote the point of interest
            # thickness, radius = 6, -1
            # keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
            # cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
            # cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)
            #
