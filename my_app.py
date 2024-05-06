# %%writefile my_app.py
import streamlit as st
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os
import cv2
import random
from matplotlib import pyplot as plt
import numpy as np


from datetime import datetime

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer


@st.cache(allow_output_mutation=True)
# def load_model():
#     print('loading model')
#     model = pickle.load(open('/kaggle/working/savedModel.sav', 'rb'))
# #     pipeline = load("/kaggle/working/savedModel.sav")
# #     return pipeline
#     im = cv2.imread("/kaggle/input/sapxuci-raw/IMG_1830.JPG")
#     outputsRaw = model(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#     #                metadata=balloon_metadata, 
#                    scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputsRaw["instances"].to("cpu"))
#     plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()
#     cv2.imwrite("output_10.JPG", out.get_image()[:, :, ::-1])
#     return outputsRaw

@st.cache(allow_output_mutation=True)
def process_image_and_get_predictions(image):
    # Process your image and get predictions here
    model = pickle.load(open('/kaggle/working/savedModel.sav', 'rb'))
    outputsRaw = model(image)  # Assuming model is already loaded
    v = Visualizer(image[:, :, ::-1],
                   metadata=MetadataCatalog.get("my_dataset_train"), 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputsRaw["instances"].to("cpu"))
    plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
    return out.get_image()
#     cv2.imwrite("output_10.JPG", out.get_image()[:, :, ::-1])
#     return outputsRaw

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to numpy array
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Process Image'):
        processed_image = process_image_and_get_predictions(image_np)
        st.image(processed_image, caption='Processed Image.', use_column_width=True)

# model = load_model()

st.title('Proof of Performance - Validation')

# review = st.text_input("What do you think about?", "This movie sucks! My whole family hates it!")
# sentiment = model.predict_proba([review])

# #st.write(sentiment)
# fig, ax = plt.subplots(figsize=(5, 5))
# sns.barplot(x=['Negative', 'Positive'], y=sentiment[0], ax=ax)

# fig.savefig("figure_name.png")
# image = Image.open('figure_name.png')
# image = Image.open("/kaggle/working/output_10.JPG")
# st.image(image)
