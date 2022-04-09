import tensorflow_hub as hub
import tensorflow as tf

def create_model():
    """Create a keras model with one \
    CNN layer transferred from MobileNet \
    which will be used as feature extractor, \
    one dense layer with 128 hidden units & \
    one output layer with 102 output units.
    
    Input image size should be (224,224,3)."""
    num_classes = 102
    image_resize = 224
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(image_resize, image_resize, 3))
    feature_extractor.trainable=False
    my_model = tf.keras.Sequential([
                feature_extractor,
                tf.keras.layers.Dense(128, activation = 'relu'),
                tf.keras.layers.Dense(num_classes, activation = 'softmax')
                ])

    my_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return my_model
