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
import os


def resize_and_show(image, name):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(name, img)


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
    DESIRED_HEIGHT = 472
    DESIRED_WIDTH = 400
    desired_ratio = DESIRED_WIDTH / DESIRED_HEIGHT
    IDEAL_INTER_IRIS_RATIO = 0.24
    IDEAL_INTER_IRIS_POSITION = {'x': 0.5, 'y': 0.45}  # (relative x,y of where center between eyes should be)
    BUFFER = {'x': 50, 'y': 50} # number of pixels to add on each side as buffer

    ext = ['png', 'jpg']
    path = 'input/'

    filenames = []
    [filenames.extend(glob.glob(path + '*.' + e)) for e in ext]

    error = []

    # create output path
    output_path = 'cropped_images'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
                print("error, no single face")
                resize_and_show(cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGBA2BGRA), filename)
                error.append(filename)
                continue

            output_image = image.numpy_view()
            iris1 = detection_result.face_landmarks[0][468]
            iris2 = detection_result.face_landmarks[0][473]
            iris_np = []

            for poi in [iris1, iris2]:
                # Draw a white dot with black border to denote the point of interest
                thickness, radius = 6, -1
                keypoint_px = _normalized_to_pixel_coordinates(poi.x, poi.y, image.width, image.height)
                iris_np.append(np.array(keypoint_px))
                # DEBUG ADD CIRCLE
                # cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
                # cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

            # place image in larger image of correct ratio
            current_ratio = image.width / image.height
            if current_ratio > desired_ratio:
                # image too wide: need to add height
                ideal_width = image.width
                ideal_height = int(image.width / desired_ratio)
                assert ideal_height > ideal_width
            else:
                ideal_height = image.height
                ideal_width = int(ideal_height * desired_ratio)
                assert ideal_height > ideal_width

            ideal_to_final_ratio = ideal_height / DESIRED_HEIGHT

            buffer_width = int(ideal_to_final_ratio * BUFFER['x'])
            buffer_height = int(ideal_to_final_ratio * BUFFER['y'])

            blank_pixel = np.uint8([0, 0, 0, 0])
            total_width = ideal_width + 2 * buffer_width
            total_height = ideal_height + 2 * buffer_height
            blank_image = np.tile(blank_pixel, (total_height, total_width, 1))
            blank_image[:output_image.shape[0], :output_image.shape[1], :output_image.shape[2]] = output_image

            # find scale for good inter iris gap
            current_gap = np.linalg.norm(iris_np[0] - iris_np[1])
            ideal_gap = ideal_width * IDEAL_INTER_IRIS_RATIO
            scaling_factor = ideal_gap / current_gap

            # find translation so that between eyes is at desired location
            inter_iris_pos = np.mean(np.array(iris_np)*scaling_factor, axis=0)
            ideal_inter_iris_pos = np.array([IDEAL_INTER_IRIS_POSITION['x']*ideal_width,
                                             IDEAL_INTER_IRIS_POSITION['y']*ideal_height])
            shift = ideal_inter_iris_pos - inter_iris_pos + np.array([buffer_width, buffer_height])

            translation_matrix = np.float32([[scaling_factor, 0, shift[0]], [0, scaling_factor, shift[1]]])
            num_rows, num_cols = blank_image.shape[:2]
            img_translation = cv2.warpAffine(blank_image, translation_matrix, (num_cols, num_rows))

            resized = cv2.resize(img_translation, (DESIRED_WIDTH + 2 * BUFFER['x'], DESIRED_HEIGHT + 2 * BUFFER['y']))
            img_BGRA = cv2.cvtColor(resized, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite("{path}/{image_name}.png".format(path=output_path, image_name=filename[6:-4]), img_BGRA)
    #         cv2.imshow("image", img_BGRA)
    #         cv2.waitKey(0)


    # cleanup
    np.savetxt(output_path+"/errors.csv", error, delimiter =", ", fmt ='% s')
    # # closing all open windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
