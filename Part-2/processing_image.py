import tensorflow as tf
import numpy as np

def process_image(img):
    """Returns a normalized image resized to
    (224, 224, 3)."""
    image_resize = 224
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (image_resize, image_resize))
    img = tf.cast(img, tf.float32)
    img /= 255
    processed_image = img.numpy()
    return processed_image