import random
import tensorflow as tf

def function(shape):

    #return  tf.Variable(tf.random.uniform([shape[0],shape[1]], minval = 0, maxval = 1,dtype=tf.dtypes.float32))
    #return tf.random.uniform([shape[0],shape[1]])
    return tf.Variable(tf.random.uniform([shape[0],shape[1]], 0, 1, dtype=tf.float32, seed=0))
