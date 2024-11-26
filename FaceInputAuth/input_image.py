# Full capture image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN


# img_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//sample_images//01.PNG"
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# h,w = img.shape[:2]
# Fct = 1000/h
# Nh = int(h*Fct)
# Nw = int(w*Fct)
# img = cv2.resize(img, (Nw,Nh))
# detector = MTCNN()
# bboxes = detector.detect_faces(img)
# try:
#     # Switching
#     if len(bboxes) == 0:
#         text = "0"
#     elif len(bboxes) > 0:
#         if len(bboxes) == 1:              # case I  : Single face
#             biggest_face = bboxes[0]
#         elif len(bboxes) > 1:             # case II : multiple face
#             max_area = 0
#             for face in bboxes:
#                 x, y, width, height = face['box']
#                 area = width*height
#                 if area > max_area:
#                     max_area = area
#                     biggest_face = face
#         # print(biggest_face)
#         bbox = biggest_face['box']
#         bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
#         landmarks = biggest_face['keypoints']
#         landmarks = np.array([landmarks["left_eye"][0],landmarks["right_eye"][0],landmarks["nose"][0],landmarks["mouth_left"][0],landmarks["mouth_right"][0],landmarks["left_eye"][1],landmarks["right_eye"][1],landmarks["nose"][1],landmarks["mouth_left"][1],landmarks["mouth_right"][1]])
#         landmarks = landmarks.reshape((2,5)).T
# except:
#     text = "Something Error"


def InputImage(img):
    # img_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//sample_images//01.PNG"
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h,w = img.shape[:2]
    # Fct = 1000/h
    # Nh = int(h*Fct)
    # Nw = int(w*Fct)
    # img = cv2.resize(img, (Nw,Nh))
    detector = MTCNN()
    bboxes = detector.detect_faces(img)
    try:
        # Switching
        if len(bboxes) == 0:
            text = "0"
        elif len(bboxes) > 0:
            if len(bboxes) == 1:              # case I  : Single face
                biggest_face = bboxes[0]
            elif len(bboxes) > 1:             # case II : multiple face
                max_area = 0
                for face in bboxes:
                    x, y, width, height = face['box']
                    area = width*height
                    if area > max_area:
                        max_area = area
                        biggest_face = face
            # print(biggest_face)
            bbox = biggest_face['box']
            bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            landmarks = biggest_face['keypoints']
            landmarks = np.array([landmarks["left_eye"][0],landmarks["right_eye"][0],landmarks["nose"][0],landmarks["mouth_left"][0],landmarks["mouth_right"][0],landmarks["left_eye"][1],landmarks["right_eye"][1],landmarks["nose"][1],landmarks["mouth_left"][1],landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2,5)).T
    except:
        text = "Something Error"

    return bbox, landmarks

# img_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//sample_images//01.PNG"
# bbox, landmarks = inputimage(img_path)
# print(bbox,landmarks)

# print("x")