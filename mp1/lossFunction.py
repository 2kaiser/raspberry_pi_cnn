import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
def function(a, X, b, y):
    A = (tf.multiply(a,(tf.matmul(X,X,transpose_a=True))))
    B =  (tf.matmul(b,X,transpose_a=True))
    Z =(tf.add(A,B)) # Z = a*(X^t*X) + b^t*X
    loss = (tf.square(Z - y))
    return loss
