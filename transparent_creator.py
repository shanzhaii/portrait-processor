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

    # Height and width that will be used by the model
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    ext = ['png', 'jpg']
    path = 'input/'
    filenames = []
    [filenames.extend(glob.glob(path + '*.' + e)) for e in ext]

    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white
    OVERLAY_COLOR = (100, 100, 0)  # cyan

    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

    # Create the options that will be used for InteractiveSegmenter
    # base_options = python.BaseOptions(model_asset_path='magic_touch.tflite')
    base_options = python.BaseOptions(model_asset_path='models/magic_touch.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=True)

    x = 0.5
    y = 0.5

    # Create the interactive segmenter
    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        for filename in filenames:
            # print(img[7:-4])
            # Create the MediaPipe image file that will be segmented
            image = mp.Image.create_from_file(filename)

            # Retrieve the masks for the segmented image
            roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                                   keypoint=NormalizedKeypoint(x, y))
            segmentation_result = segmenter.segment(image, roi)
            category_mask = segmentation_result.confidence_masks[0]

            image_data = image.numpy_view()
            # Generate solid color images for showing the output segmentation mask.
            # fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            # fg_image[:] = MASK_COLOR
            # bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            #
            # condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
            # output_image = np.where(condition, fg_image, bg_image)
            mask = np.stack((category_mask.numpy_view(),), axis=-1)*255
            mask_int = mask.astype(np.uint8)
            cv_image = cv2.imread(filename)
            b_channel, g_channel, r_channel = cv2.split(cv_image)
            if mask_int.shape[:-1] != b_channel.shape:
                mask_int = np.swapaxes(mask_int, 0, 1)
            img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask_int))
            cv2.imwrite("output/{image_name}.png".format(image_name=filename[6:-4]), img_BGRA)

            # # Draw a white dot with black border to denote the point of interest
            # thickness, radius = 6, -1
            # keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
            # cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
            # cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

