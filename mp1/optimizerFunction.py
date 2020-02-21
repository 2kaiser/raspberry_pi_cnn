import tensorflow as tf
def function(loss,lr):
    adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    return adam.minimize(loss)#adam.minimize(loss,var_list = X)
