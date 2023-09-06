import cv2
import mediapipe as mp
from PIL import Image, ImageOps
import math
import glob
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
from face_detector import IDEAL_INTER_IRIS_POSITION, DESIRED_HEIGHT, DESIRED_WIDTH, BUFFER
import os


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


if __name__ == "__main__":
    # Height and width that will be used by the model

    ext = ['png', 'jpg']
    path = 'output_cropped_images/'

    filenames = []
    [filenames.extend(glob.glob(path + '*.' + e)) for e in ext]

    # create output path
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # magictouch clic location: between eyes
    x = (DESIRED_WIDTH * IDEAL_INTER_IRIS_POSITION['x'] + BUFFER['x']) / (DESIRED_WIDTH + 2 * BUFFER['x'])
    y = (DESIRED_HEIGHT * IDEAL_INTER_IRIS_POSITION['y'] + BUFFER['y']) / (DESIRED_HEIGHT + 2 * BUFFER['y'])

    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

    # Create the options that will be used for InteractiveSegmenter
    # base_options = python.BaseOptions(model_asset_path='magic_touch.tflite')
    base_options = python.BaseOptions(model_asset_path='models/magic_touch.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=True)

    # Create the interactive segmenter
    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        for filename in filenames:
            # print(img[7:-4])
            # Create the MediaPipe image file that will be segmented
            # using PIL as it reads EXIF properly
            pil_img = Image.open(filename)
            # img_cv = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
            pil_img = ImageOps.exif_transpose(pil_img)
            img_array = np.asarray(pil_img.convert('RGBA'))
            # treat as image with transparency or not
            image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=img_array)

            # Retrieve the masks for the segmented image
            roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                                   keypoint=NormalizedKeypoint(x, y))
            segmentation_result = segmenter.segment(image, roi)
            category_mask = segmentation_result.confidence_masks[0]

            image_data = image.numpy_view()
            mask = np.stack((category_mask.numpy_view(),), axis=-1) * 255
            mask_int = mask.astype(np.uint8)
            cv_image = cv2.imread(filename)
            b_channel, g_channel, r_channel = cv2.split(cv_image)
            if mask_int.shape[:-1] != b_channel.shape:
                mask_int = np.swapaxes(mask_int, 0, 1)
            img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask_int))
            image_name = filename.split("\\")[1].split(".")[0]
            cv2.imwrite("{path}/{image_name}.png".format(path=output_path, image_name=image_name), img_BGRA)
