# Visual_analytics_Final_assignment
## ------ SCRIPT DESCRIPTION ------
This repository contains two scripts aiming to try and predict whether a movie is good or bad (according to IMDB-score) based on its poster.

In practice this repo contains two scripts and a pretrained model. One to create said model and one to predict images with the pretrained model. This was done so you wouldn't have to create the model each time you wished to predict a movies poster, but instead just could use the already created model.

NOTE: Due to file size limitations, the **pretrained model is in a .zip file and needs to be unziped before use.**

## ------ METHODS ------
Originally the script was supposed to be a multiclass classification using VGG16 split into 5 classes: Terrible, Bad, Decent, Good and Great. However, it soon became apparent that not only is it increasingly difficult to predict a movie's quality based on its poster (if not impossible), the data was also heavily unbalanced towards the category "Good". It was therefore decided to instead try creating a binary classification using VGG16, with all movies with an IMDB-score of =<7 being deemed good and >7 being deemed bad. This cut follows the average movie score according to IMDB. This means that any movie in the "good" data are movies with an above-average score.

As you will see with the model, it is not good at predicting a movie's quality. Even after countless hours trying to optimize the model, it seldomly gave a validation accuracy of above 0.6. These optimisations included switching from a SGD optimiser to an "adam" optimiser, introducing a batch normalization layer and using image augmentation and using different epochs and batch sizes.  

## ------ DATA ------
The data is from the kaggle dataset: https://www.kaggle.com/datasets/phiitm/movie-posters. 

The data was split into training and a testing data of similar size (80-20), split into two classes: "Bad" and "Good". Images were then randomly removed from the "Bad" data to create a balanced dataset of 5.840 images.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py scripts to create the image classification model and to predict images.
- The precreated / pretrained model created from the model creation script

"in" FOLDER:
- This is where the data used in the scripts should be placed. In other words this is where the movie posters train and test data should be placed.
- Any posters that you wish to predict using the poster_prediction.py script should be placed in the "Prediction_images" folder.

"out" FOLDER:
- This is where the model and its history plot will be saved

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
- The model_creation.py script requires you to give the arguments "-e" / "--epoch" (how many epochs you want it to train) and "-b" / "--batch" (for batchsize). The pretrained model was created with 10 epochs and batchsize 128

- The poster_prediction.py script requires you to give the argument "-i" / "--image" (the name of the image you want to predict)

## ------ RESULTS ------
Even with the different optimisations of the model, it did not become good at predicting whether or not a movie is good or bad based on its poster. There is almost as if there is a 50 / 50 chance of it guessing correctly or wrong. Notable critically appraised movies such as Batman (2022) or Dune were categorized as bad.  

In short: a model can't predict what this model is trying to predict. This follows in line with other models that have tried to do the same thing (such as https://www.kaggle.com/code/phiitm/can-we-judge-a-movie-by-it-s-poster).
