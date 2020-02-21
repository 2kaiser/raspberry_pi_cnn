
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import requests
import gzip
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json
import secrets
###################################################################################################################################################
# later...
# load json and create model
json_file = open('lab1_keras_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
###################################################################################################################################################
#Step 1: Raspberry Pi sends a POST request to remote server for input images.
r = requests.post("https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php", data={'netid': 'skcheun2', 'request': 'testdata'},allow_redirects=True)
print(r.status_code, r.reason)
###################################################################################################################################################
#Step 2: Remote server responds with images to classify.
def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data
filename = r.url.split("/")[-1]
testset_id = filename.split(".")[0].split("_")[-1]
with open(filename, 'wb') as f:
    f.write(r.content)
data = load_dataset(filename), testset_id
print(len((data[0])))
test_Data = [a.reshape(28,28,1) for a in data[0]]#img = Image.fromarray(data[0], 'RGB')
test_Data = np.asarray(test_Data, dtype=np.float32)
#normalize data
test_Data = test_Data.astype('float32') / 255
print("test_Data size")
print(((test_Data.shape)))
###################################################################################################################################################
#setp 3 setup premade model
loaded_model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
y_test = np.zeros(1000)
#predict test_labels
prediction = loaded_model.predict(test_Data)
print(len(prediction))
res = ''
    
for pred in prediction:
    for i in range(10):
        if pred[i] > .2:
            res = res + str(i)
            break
print(res)
###################################################################################################################################################
#Step 4: Raspberry Pi sends the second POST request with inference results to remote server for result verification.
###################################################################################################################################################
r = requests.post("https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php", data={'request': 'verify','netid': 'skcheun2','prediction': res,'testset_id': testset_id},allow_redirects=True)
###################################################################################################################################################
#Step 5: Remote server compares the prediction results with the correct labels, and responds with number of correct predictions.
print("status code")
print(r.status_code, r.reason)
print("NUMBER OF CORERECT DATA POINTS")
print(r.text)
###################################################################################################################################################
