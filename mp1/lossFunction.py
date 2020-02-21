import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
def function(a, X, b, y):

    A = (tf.multiply(a,(tf.matmul(X,X,transpose_a=True))))
    B =  (tf.matmul(b,X,transpose_a=True))
    Z =(tf.add(A,B)) # Z = a*(X^t*X) + b^t*X
    #(Z - y)**2
    #error = tf.Variable(tf.subtract(tf.multiply(a,(tf.matmul(X,X,transpose_a=True))),tf.matmul(b,X,transpose_a=True)) - y,name='error')
    loss = (tf.square(Z - y))
    #loss = tf.Graph()
    #with loss.as_default():
  # Define operations and tensors in `g`.
        #hello = tf.constant('hello')
        #assert hello.graph is g
    #a = tf.placeholders(tf.float32, []);
    #loss = tf.Variable(tf.square(tf.add(tf.multiply(a,(tf.matmul(X,X,transpose_a=True))),tf.matmul(b,X,transpose_a=True)) - y),"loss")
        #assert result.graph is loss
    #loss = tf.Variable(tf.square(tf.add(tf.multiply(a,(tf.matmul(X,X,transpose_a=True))),tf.matmul(b,X,transpose_a=True)) - y),"loss")
    #function = tf.matmul(X,X,transpose_a=True)

    return loss

    #return model
