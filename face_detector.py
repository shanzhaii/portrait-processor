from PIL import Image, ImageOps
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import glob
import math

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('image', img)

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
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    ext = ['png', 'jpg']
    path = 'input/'
    filenames = []
    [filenames.extend(glob.glob(path + '*.' + e)) for e in ext]

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for filename in filenames:
            # STEP 3: Load the input image.
            # using PIL as it reads EXIF properly
            pil_img = Image.open(filename)
            # img_cv = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
            pil_img = ImageOps.exif_transpose(pil_img)
            img_array = np.asarray(pil_img.convert('RGBA'))
            # treat as image with transparency or not
            image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=img_array)

            # STEP 4: Detect face landmarks from the input image.
            detection_result = detector.detect(image)

            # STEP 5: Process the detection result. In this case, visualize it.
            # annotated_image = draw_landmarks_on_image(image.numpy_view()[:,:,:3], detection_result) # can only draw on rgb images
            if len(detection_result.face_landmarks) != 1:
                print("error, not single face")
                resize_and_show(cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                continue

            output_image = image.numpy_view()
            iris1 = detection_result.face_landmarks[0][468]
            iris2 = detection_result.face_landmarks[0][473]

            for poi in [iris1, iris2]:
                # Draw a white dot with black border to denote the point of interest
                thickness, radius = 6, -1
                keypoint_px = _normalized_to_pixel_coordinates(poi.x, poi.y, image.width, image.height)
                cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
                cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

            resize_and_show(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
