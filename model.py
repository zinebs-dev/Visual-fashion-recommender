import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Modele global
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def process_image(img_path):
    # OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Keras
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

def get_features(img_path):
    img_array = process_image(img_path)
    features = model.predict(img_array, verbose=0)
    return features.flatten()