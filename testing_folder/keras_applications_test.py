from keras.applications.vgg16 import VGG16
from keras.layers import Input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import numpy as np

import cv2

test_image_path = "00054_gun_wielder.jpg"

test_image = cv2.imread(test_image_path)

test_input = np.array([test_image])

n_classes = 2

# create the base pre-trained model
vis_input = Input(shape=(test_image.shape[0], test_image.shape[1], test_image.shape[2]), name="absolute_input")

base_model = VGG16(input_tensor=vis_input, weights='imagenet',input_shape=(test_image.shape[0], test_image.shape[1], test_image.shape[2]), include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


prediction = model.predict(test_input)

print(prediction)