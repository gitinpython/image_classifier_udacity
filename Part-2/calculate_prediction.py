from PIL import Image
from processing_image import process_image
from creating_model import create_model
import numpy as np
import tensorflow as tf

def map_predicted_prob_vs_label(prediction):
    """Returns a map of probability vs label pair."""
    predicted_prob_and_label = {k:v for v,k in enumerate(prediction)}
    highest_prob_label = dict()
    for prob in sorted(predicted_prob_and_label, reverse=True):
        highest_prob_label[prob] = predicted_prob_and_label[prob]
    
    return highest_prob_label

def predict(image_path=None, model_path=None, top_k=None):
    """Returns the image classes with associated probabilities \
       in decreasing order.
       
       Read the image path & process it. Use the saved trained keras model \
       to load the weights on the duplicate keras model created. Once, the weights \
       are loaded, use this model for prediction on the input image.
       
       Unless "top_k" is specified, this will return all the classes & associated \
       probabilities in the decreasing order.
       """
    if image_path is None:
        raise Exception("Must provide an image path!")
    
    if model_path is None:
        raise Exception("Must load a keras model in .h5 format to run the classifier!")
    
    # create duplicate keras model (this should be exactly similar to the trained keras model)
    # Load the wights on this duplicate model from the saved trained keras model (.h5 file)
    model = create_model()
    model.load_weights(model_path)
    
    #load image in the form of numpy
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    #pre-process image: resize & normalize pixels
    processed_test_image = process_image(test_image)
    modified_processed_test_image = np.expand_dims(processed_test_image, axis=0)
    
    #convert to tensor before predicting by model
    processed_img = tf.convert_to_tensor(modified_processed_test_image)
    predicted_prob = model.predict(processed_img) #returns 102 probabilities
    
    #map probabilities with labels & sort them in the order of highest probabilities
    sorted_prob_vs_label = map_predicted_prob_vs_label(predicted_prob[0]) #default contains only one image
    
    #Get top_k list of probabilities & their respective labels
    top_k_probs = list(sorted_prob_vs_label.keys())[:top_k]
    top_k_labels = [str(i+1) for i in list(sorted_prob_vs_label.values())[:top_k]]
    
    return top_k_probs, top_k_labels