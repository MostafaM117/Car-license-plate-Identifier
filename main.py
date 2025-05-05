import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import easyocr
from shapely.geometry import Polygon
from ultralytics import YOLO
import tempfile
import os

# torch.serialization.add_safe_globals([torch.nn.Module, 'ultralytics.nn.tasks.OBBModel'])
# model = YOLO('d:\AI_Final_Project\yolov8n-obb.pt')
# model.eval()

st.title('Car License Identifier')
img_upload = st.file_uploader('Click Browse files to upload an image', type=["jpg", "jpeg", "png"])
st.caption('Upload an image of a vehicle with numbers plate, and click :blue[Detect License Plate] button.')

# split page 
col1, col2 = st.columns(2)

@st.cache_resource
def load_model():
  model = YOLO('d:\AI_Final_Project\yolov8n-obb.pt')
  return model

model = load_model()

result = []
if img_upload is not None:
  # image = Image.open(img_upload)
  # image = np.array(image)
  file_bytes = np.asarray(bytearray(img_upload.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  orig_img = img.copy()
  st.image(Image.open(img_upload), caption = 'Uploaded Image', use_container_width=True)
  # with torch.no_grad():
  #   predictions = model(image)
  # st.image(image, caption='Uploaded Image', use_column_width=True)
  # st.write(predictions)

  with col1:

    if st.button('Detect License Plate'):
      # model = YOLO('d:\AI_Final_Project\yolov8n-obb.pt')
      first_result = model(img)[0]

      if first_result.obb is None or len(first_result.obb.xywhr) == 0:
        st.error('No license plate detected by the model.')
      else: 
        obbs = first_result.obb.xyxy.cpu().numpy().astype(int)
        class_ids = first_result.obb.cls.cpu().numpy().astype(int)
        names = first_result.names
        # label_id = int(first_result.obb.cls[0])
        # label = names[label_id]
        found_plate = False
        for i, box in enumerate(obbs):
          class_name = names[class_ids[i]]
          if class_name.lower() == 'plate':
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
              plate_img = img[y1:y2, x1:x2]
              if plate_img.size == 0:
                continue
        # if plate_img.size == 0:
        #   st.error('Detected bounding box is empty.')
        # else:
              st.image(plate_img, caption='Detected Plate Region', use_container_width=True)
          # OCR
              plate_grey  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
              plate_resized   = cv2.resize(plate_grey,  None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
              reader = easyocr.Reader(['ar','en'], gpu=False)
              result = reader.readtext(plate_resized)

#         # old code
#         # plate_img = cv2.convertScaleAbs(plate_img, alpha= 2.0, beta= 0)
#         result = reader.readtext(plate_img)

              with col2:
                st.image(plate_resized, caption="Detected Plate Region", use_container_width=True)
                if result:
                  text = result[0][-2]
                  # img_boxed = img.copy()
                  cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), caption="Final Result", use_container_width=True)
#             # Not Precise Text Detection
                  st.success(f"Detected License Plate: `{text}`")
      
                else:
                  st.warning("No text detected from OCR")
              found_plate = True
              break
else:
    st.info("Upload an image to detect its plate numbers")