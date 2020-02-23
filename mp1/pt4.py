import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print((train_images.shape))
#normalize data
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
# reshape dataset to have a single channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
model = keras.Sequential()
#add model layers
model.add(tf.keras.layers.Conv2D(filters=3, strides = 1,kernel_size=5, padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=3, strides = 1, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# return the constructed network architecture
#compile model using accuracy to measure model performance
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#train the model
history = model.fit(train_images, train_labels, batch_size=64, epochs=7)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
print(history.history['loss'])
# "Loss"
plt.scatter(range(len(history.history['loss'])),history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("lab1_keras_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
