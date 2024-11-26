import time

from input_image import InputImage
from crop_face import FacePreprocessor
import cv2
import matplotlib.pyplot as plt
from face_embedding import FaceModel

# img_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//sample_images//01.PNG"

# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# h,w = img.shape[:2]
# Fct = 1000/h
# Nh = int(h*Fct)
# Nw = int(w*Fct)
# img = cv2.resize(img, (Nw,Nh))

# bbox, landmarks = InputImage(img)

# preprocessor = FacePreprocessor(image_size='112,112',margin=44)
# nimg = preprocessor.preprocess(img,bbox,landmarks)




# embedding_model_path = "C://Users//Acer//Desktop//Cocoeye_cluster//FaceInputAuth//models//w600k_r50.onnx"
# face_model = FaceModel()

# prep_img = face_model.preprocess_image(nimg)
# embedding = face_model.get_embedding(prep_img).reshape(1,-1)

# print(embedding)

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2


app = FastAPI()

@app.post("/api/v2/facescan/upload")
async def create_upload_file(
    file : UploadFile = File(...)
):
    print("x"*1000)
    # input image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_shape = image.shape
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w = img.shape[:2]
    Fct = 1000/h
    Nh = int(h*Fct)
    Nw = int(w*Fct)
    img = cv2.resize(img, (Nw,Nh))


    bbox, landmarks = InputImage(img)


    preprocessor = FacePreprocessor(image_size='112,112',margin=44)
    nimg = preprocessor.preprocess(img,bbox,landmarks)

    # ช้า
    st = time.time()
    face_model = FaceModel()
    ed = time.time()                                                    #                 
    print("01:",ed-st)
    
    prep_img = face_model.preprocess_image(nimg)
    embedding = face_model.get_embedding(prep_img).reshape(1,-1)


    print("x"*1000)
    result = {
        "Embedding_vector": embedding.tolist()
    }
    return result
