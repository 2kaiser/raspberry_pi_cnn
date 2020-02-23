import random
import tensorflow as tf
import numpy as np
def function(shape):
    return tf.Variable(np.random.rand(shape[0],shape[1]).astype(np.float32))
