from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import backend as K
########################################################################################################################
#load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print((train_images.shape))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
########################################################################################################################
#create model
model = models.Sequential()
model.add(layers.Conv2D(3, (5, 5), padding = "VALID",activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=2,strides=2))
model.add(layers.Conv2D(3, (3, 3), activation='relu',padding = "SAME", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=2,strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
########################################################################################################################
#run the model
history = model.fit(train_images, train_labels, epochs=10, batch_size =32,
                    validation_data=(test_images, test_labels))
########################################################################################################################
#plot results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
########################################################################################################################
# serialize model to JSON
model_json = model.to_json()
with open("lab1_lowlevel_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
