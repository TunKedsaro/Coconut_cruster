# Face Embedding
import onnxruntime as ort
import cv2
import numpy as np
# import boto3
# import aws_keys
import time
class FaceModel:
    def __init__(self):
        st = time.time()
        self.session = ort.InferenceSession("/app/models/w600k_r50.onnx")
        # s3 = boto3.client(
        #     "s3",
        #     region_name           = aws_keys.AWS_DEFAULT_REGION,
        #     aws_access_key_id     = aws_keys.AWS_ACCESS_KEY_ID,
        #     aws_secret_access_key = aws_keys.AWS_SECRET_ACCESS_KEY
        #     )
        # # w600k_r50.onnx
        # response = s3.get_object(
        #     Bucket = "model-test-211124-0920",
        #     Key = "w600k_r50.onnx"
        # )
        # face_embedding_Bytes = response['Body'].read()
        # self.session = ort.InferenceSession(face_embedding_Bytes)
        ed = time.time()
        print("load_model from s3:",ed-st)

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

# embedding_model_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//models//w600k_r50.onnx"
# face_model = FaceModel(embedding_model_path)
# prep_img = face_model.preprocess_image(nimg)
# # embedding = face_model.get_embedding(prep_img).reshape(1,-1)
# # print(embedding)
