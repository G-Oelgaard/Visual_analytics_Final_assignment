import os, sys
sys.path.append(os.path.join(".."))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
                
from tensorflow.keras.applications.vgg16 import (preprocess_input)
                
import numpy as np

import argparse
                
## Functions ##
                
# Load model
def load_model():
    filepath = os.path.join("poster_model.h5")
    model = tf.keras.models.load_model(filepath)
    
    return model

# Predict images
def image_prediction(model,image):
    label_names = ["Bad", "Good"]
    
    pred_img = load_img(os.path.join("..","in","Prediction_images",image), target_size = (224, 224))
    pred_img = img_to_array(pred_img)
    pred_img = pred_img.reshape((1,pred_img.shape[0],pred_img.shape[1],pred_img.shape[2]))
    
    pred_img = preprocess_input(pred_img)
    
    prediction = model.predict(pred_img)
    
    print(f"Result:\n The model predicts that this movie is {label_names[np.argmax(prediction)]}")
    
# Args_parse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help="What image you want the model to predict")
    args = vars(ap.parse_args())
    return args

## Main ##
# Defining main
def main():
    args = parse_args()
    model = load_model()
    image_prediction(model, args["image"])
              
# Running main
if __name__ == "__main__":
    main()