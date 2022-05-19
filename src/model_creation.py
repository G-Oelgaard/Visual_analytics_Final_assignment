## Importing packages ##
# base tools

import os, sys
sys.path.append(os.path.join(".."))

# Sklearn
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)

from tensorflow.keras.models import Model

# for plotting
import numpy as np
import matplotlib.pyplot as plt

#args_parse
import argparse

## functions ##

# Loading data
def load_data(batch):
    
    train_data = keras.utils.image_dataset_from_directory(
        directory= "../in/Train",
        labels='inferred',
        label_mode='binary',
        batch_size=int(batch),
        image_size=(224, 224))

    test_data = keras.utils.image_dataset_from_directory(
        directory= "../in/Test",
        labels='inferred',
        label_mode='binary',
        batch_size=int(batch),
        image_size=(224, 224))
    
    #train = ImageDataGenerator(rescale=1/255, horizontal_flip=True, rotation_range=20)
    #test = ImageDataGenerator(rescale=1/255, horizontal_flip=True, rotation_range=20)

    #train_data = train.flow_from_directory(os.path.join("..","Input","Good_or_bad_balanced","Train"),
                                              #target_size=(224,224),
                                              #batch_size = int(batch),
                                              #class_mode = 'binary')

    #test_data = test.flow_from_directory(os.path.join("..","Input","Good_or_bad_balanced","Test"),
                                              #target_size=(224,224),
                                              #batch_size = int(batch),
                                              #class_mode = 'binary')
    
    label_names = train_data.class_names
    
    return label_names, test_data, train_data

# Load VGG16, add bottom layers and run
def load_model(train_data, test_data, epoch, batch):
    tf.keras.backend.clear_session() # just to be sure nothing funky happens
    
    model = VGG16(include_top = False,
             pooling = "avg",
             input_shape = (224,224,3))
    
    for layer in model.layers:
        layer.trainable = False
    
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, activation='relu')(bn)
    class2 = Dense(128, activation="relu")(flat1)
    output = Dense(1, activation="sigmoid")(class1)

    model = Model(inputs = model.inputs,
                  outputs = output)

    model.summary()
    
    model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

    H = model.fit(train_data,
                    validation_data = test_data,
                    epochs = int(epoch),
                    batch_size = int(batch),
                    verbose = 1)
    
    return H, model

# save history plots
def save_history_plots(H, epoch):
    
    outpath = os.path.join("..","out", "poster_pred_loss_accuracy.jpg")

    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, int(epoch)), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, int(epoch)), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, int(epoch)), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, int(epoch)), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.savefig(outpath)

    plt.show()

# Plot and save classification report
def class_report(test_data, model, label_names):
    y_test = np.concatenate([y for x, y in test_data], axis=0)
    X_test = np.concatenate([x for x, y in test_data], axis=0)
    predictions = model.predict(X_test, batch_size=32)
    preds = [1 if i>0.5 else 0 for i in predictions]
    print(classification_report(y_test, preds, target_names = label_names))
    
    outpath = os.path.join("..","out","poster_pred_class_report.txt")
    
    with open(outpath,"w") as file:
        file.write(str(class_report))
        
# save model
def model_save(model):
    outpath = os.path.join("..","out","poster_model.h5")
    
    model.save(outpath)
        
# args_parse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epoch", required = True, help="How many epochs the model should run.")
    ap.add_argument("-b", "--batch", required = True, help="Define the batch size the model should use.")
    args = vars(ap.parse_args())
    return args

## Main ##
# Defining main
def main():
    args = parse_args()
    label_names, test_data, train_data = load_data(args["batch"])
    H, model = load_model(train_data, test_data, args["epoch"], args["batch"])
    save_history_plots(H, args["epoch"])
    class_report(test_data, model, label_names)
    model_save(model)
              
# Running main
if __name__ == "__main__":
    main()
