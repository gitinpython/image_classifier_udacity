# image_classifier_udacity
Create Image Classifier - Intro to ML with Tensor FLow, Udacity Nanodegree Program

This project is associalted with the lesson - Deep Learning (with tensorflow) from the Udacity's Nanodegree program: "Introduction to Machine Learning with Tensorflow".
Here, we will train an image classifier to recognize different species of flowers. We will need to develop a Deep Learning Keras model using tensorflow application & use the [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories for our training. Then, we will need to develop a Command Line Interface (CLI) script that uses the trained network to predict the class for an input image. 

Following files & folders are available, described as below:

- **Part-1**
  - Contains a jupyter notebook (.ipynb) file & HTML file
  - The following has been done in this notebook:
    - Dataset download & normalizing the image into desired size & pixel range
    - Creating a pipeline, split into training. validation & testing datasets
    - Building a keras model, training the classifier & drawing predictions on the testing dataset
    - Saving the best trained model in a .h5 file extension using Early Stopping
    - Creating an untrained keras model which is exactly similar to the trained model & loading the weights from the saved model (in .h5 file)
    - Using this duplicate model with loaded weights to predict on the test images
   - Utility functions: 
     - workspace_utils.py: Helper functions that include an iterator wrapper called keep_awake and a context manager called active_session that can be used to maintain an active session during long-running processes.
     
- **Part-2**
  - Contains the CLI script - "predict.py" along with several utility scripts
  - Utility scripts:
    - processing_image.py: Processes the input image, normalizes & resizes
    - creating_model.py: Creates an untrained keras model which is exactly similar to the trained model so that we can use to load the weights from the saved model (.h5 file)
    - calculate_prediction.py: This will draw prediction on the input image & returns the image classes with associated probabilities in decreasing order
    - label_map.json: JSON file mapping labels to flower names
   - CLI script aka predict.py: 
     - This will take the input image & the saved keras model as basic inputs & return the most likely image class & its associated probability
     - It will also take optional arguments such as:
       - to print the top K image classes & associated probabilities
       - to print the category names of the flower species corresponding to the predicted image classes
       
- **test_images** : Folder contains 4 test images to check the prediction module
- **best_model.h5** : Saved Keras Model from Part-1




References used for code:
- https://www.tensorflow.org/datasets/overview
- https://github.com/tensorflow/datasets/issues/1977
- https://github.com/tensorflow/datasets/issues/1998 
- https://www.tensorflow.org/datasets/splits
- https://towardsdatascience.com/checkpointing-deep-learning-models-in-keras-a652570b8de6
