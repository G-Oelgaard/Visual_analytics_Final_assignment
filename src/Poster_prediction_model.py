## Importing packages ##
# base tools
import os, sys
sys.path.append(os.path.join(".."))

# Sklearn
from sklearn.metrics import classification_report

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
    
    train = ImageDataGenerator(rescale=1/255, horizontal_flip=True, rotation_range=20)
    test = ImageDataGenerator(rescale=1/255, horizontal_flip=True, rotation_range=20)

    train_data = train.flow_from_directory(os.path.join("..","in","Train"),
                                              target_size=(224,224),
                                              batch_size = int(batch),
                                              class_mode = 'categorical')

    test_data = test.flow_from_directory(os.path.join("..","in","Test"),
                                              target_size=(224,224),
                                              batch_size = int(batch),
                                              class_mode = 'categorical')
    
    label_names = ["Bad", "Good"]
    
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
    output = Dense(2, activation="sigmoid")(class1)

    model = Model(inputs = model.inputs,
                  outputs = output)

    model.summary()
    
    model.compile(optimizer="adam",
              loss="categorical_crossentropy",
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
def classification_report(model, test_data, batch, label_names):
    Y_pred = model.predict(test_data, 1320 // int(batch)+1)
    y_pred = np.argmax(Y_pred, axis=1)
    class_report = classification_report(test_data.classes, y_pred, target_names=label_names)
    print(class_report)
    
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
    #classification_report(model, test_data, label_names, args["epoch"])
    model_save(model)
              
# Running main
if __name__ == "__main__":
    main()
    
