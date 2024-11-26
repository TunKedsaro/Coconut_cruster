# Full capture image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN

img_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//sample_images//01.PNG"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]
Fct = 1000/h
Nh = int(h*Fct)
Nw = int(w*Fct)
img = cv2.resize(img, (Nw,Nh))
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

# print(bbox)
# print(landmarks)

# Crop face
from skimage import transform as trans
class FacePreprocessor:
    def __init__(self,image_size='112,112',margin=44):
        self.image_size = [int(x) for x in image_size.split(',')]
        if len(self.image_size) == 1:
            self.image_size = [self.image_size[0],self.image_size[0]]
        self.margin = margin
        assert len(self.image_size) == 2
        assert self.image_size[0] == 112 and (self.image_size[1] == 112 or self.image_size[1] == 96)
    def read_image(self,img_path,mode='rgb',layout='HWC'):
        if mode == 'gray': # gray -> gray
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if mode == 'rgb':
                img = img[..., ::-1]
            if layout == 'CHW':
                img = np.transpose(img,(2,0,1))
        return img
    def preprocess(self, img, bbox=None, landmark=None):
        if isinstance(img, str):
            img = self.read_image(img)
        M = None
        if landmark is not None:
            assert len(self.image_size) == 2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)

            if self.image_size[1] == 112:
                src[:, 0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]

        if M is None:
            return self._center_crop(img, bbox)
        else:
            return self._warp_image(img, M)

    def _center_crop(self, img, bbox):
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - self.margin // 2, 0)
        bb[1] = np.maximum(det[1] - self.margin // 2, 0)
        bb[2] = np.minimum(det[2] + self.margin // 2, img.shape[1])
        bb[3] = np.minimum(det[3] + self.margin // 2, img.shape[0])

        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(self.image_size) > 0:
            ret = cv2.resize(ret, (self.image_size[1], self.image_size[0]))
        return ret

    def _warp_image(self, img, M):
        assert len(self.image_size) == 2
        warped = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        return warped

preprocessor = FacePreprocessor(image_size='112,112',margin=44)
nimg = preprocessor.preprocess(img,bbox,landmarks)
# plt.imshow(nimg)
# plt.show()


# Face Embedding
import onnxruntime as ort
class FaceModel:
    def __init__(self, embedding_model_path):
        self.session = ort.InferenceSession(embedding_model_path)

    def preprocess_image(self, img_input):
        # Check if the input is a file path (string) or a NumPy array (image)
        if isinstance(img_input, str):  # If it's a file path
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Image not found at path: {img_input}")
        elif isinstance(img_input, np.ndarray):  # If it's an image (NumPy array)
            img = img_input
        else:
            raise TypeError("Input must be a file path (str) or an image (np.ndarray)")

        # Resize the image to (112, 112)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def get_embedding(self, img):
        inputs = self.session.get_inputs()
        outputs = self.session.run(None, {inputs[0].name: img})
        embedding = outputs[0]
        return embedding[0]

embedding_model_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//models//w600k_r50.onnx"
face_model = FaceModel(embedding_model_path)
prep_img = face_model.preprocess_image(nimg)
embedding = face_model.get_embedding(prep_img).reshape(1,-1)
print(embedding)