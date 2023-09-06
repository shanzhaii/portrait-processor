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

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    if len(face_landmarks_list) != 1:
        print("error, not single face")
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

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
            annotated_image = draw_landmarks_on_image(image.numpy_view()[:,:,:3], detection_result) # can only draw on rgb images
            resize_and_show(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
