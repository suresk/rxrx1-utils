from functools import partial
import tensorflow as tf

def get_transforms():
    return [partial(tf.image.flip_up_down)]