import numpy as np

import os
import sys

def get_dir_path(dir_name):
    base_path = os.getcwd().split("/")
    while base_path[-1] != str(dir_name):
        base_path = base_path[:-1]
    return "/".join(base_path) + "/"

models_path = os.path.join(get_dir_path("p5_afm_2018_demo"),"models")
folders = os.listdir(models_path)
for folder in folders:
    folder_path = os.path.join(models_path, folder)
    if os.path.isdir(folder_path) and not folder_path in sys.path:
        sys.path.append(folder_path)


import keras
from keras.datasets import mnist
from keras.layers import InputLayer
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json

import tensorflow as tf
from keras.backend import set_session

import random

class KerasCNN(object):
    """A simple CNN model implemented using the Tensorflow estimator API"""

    def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir,
                 additional_args={}):
        super(KerasCNN, self).__init__()
        self.model_input_dim_height = model_input_dim_height
        self.model_input_dim_width = model_input_dim_width
        self.model_input_channels = model_input_channels
        model_input_dim = [model_input_dim_height, model_input_dim_width, model_input_channels]
        self.input_ = tf.placeholder(
                tf.float32,
                shape = [None] + list(model_input_dim),
                name = "in_"
        )
        self.labels_ = tf.placeholder(
            name="lbl_",
            shape=[None, 1],
            dtype=tf.float32
        )
        self.n_classes = n_classes
        self.model_dir = model_dir
        self.sess = tf.Session()

        # model specific variables
        # Training Parameters

        if ("learning_rate" in additional_args):
            self.learning_rate = additional_args["learning_rate"]
        else:
            self.learning_rate = 0.001
            print("using default of " + str(self.learning_rate) + " for " + "learning_rate")

        if ("dropout" in additional_args):
            self.dropout = additional_args["dropout"]
        else:
            self.dropout = 0.25
            print("using default of " + str(self.dropout) + " for " + "dropout")

        self.model = None
        self.InitaliseModel(model_dir=self.model_dir)

    ### Required Model Functions
    def InitaliseModel(self, model_dir="model_dir"):
        opts = tf.GPUOptions(allow_growth=True)
        conf = tf.ConfigProto(gpu_options=opts)
        # trainingConfig = tf.estimator.RunConfig(session_config=conf)
        self.sess = tf.Session(config=conf)
        keras.backend.set_session(self.sess.run)

        self.model = self.BuildModel(self.model_input_dim_height, self.model_input_dim_width, self.model_input_channels, self.n_classes,self.dropout)
        self.logits = self.model(self.input_)
        self.sess.run(tf.global_variables_initializer())
        self.loss = keras.losses.categorical_crossentropy(self.logits,self.labels_)

    def TrainModel(self, train_x, train_y, batch_size, num_steps, val_x= None, val_y=None):
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        while train_y != [] and num_steps != 0:
        #     Get batch
            batch_x, batch_y, train_x, train_y = self._get_batches(train_x, train_y, batch_size)
            feed_dict = {
                self.input_ : batch_x,
                self.labels_ : batch_y
            }

            self.sess.run(train_step, feed_dict)
            num_steps -= 1

        """
        if (type(train_x) != dict):
            input_dict = {"input": train_x}
        else:
            input_dict = train_x

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=self.learning_rate,),
              target_tensors = [self.labels_],
              metrics=['accuracy'])

        if(val_x is not None and val_y is not None):
            self.model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=num_steps,
              verbose=1,
              validation_data=(val_x, val_y))
        else:
            self.model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=num_steps,
          verbose=1)
        """

    def EvaluateModel(self, eval_x, eval_y, batch_size):
        if (type(eval_x) != dict):
            input_dict = {"input": eval_x}
        else:
            input_dict = eval_x

        # Train the Model
        # return self.model.evaluate(eval_x,eval_y, batch_size=batch_size)
        loss = []
        while eval_y.size != 0 and num_steps != 0:
            batch_x, batch_y, eval_x, eval_y = self._get_batches(eval_y,eval_y, batch_size)
            feed_dict = {
                self.input_: batch_x,
                self.labels_: batch_y
            }
            loss.append(self.sess.run(self.loss, feed_dict))
            num_steps -= 1
        return sum(loss)/len(loss)
    def Predict(self, predict_x):
        if (type(predict_x) != dict):
            input_dict = {"input": predict_x}
        else:
            input_dict = predict_x

        feed_dict = {
            self.input_: predict_x
        }
        predictions = self.sess.run(self.logits, feed_dict)
        print("[np.argmax(prediction) for prediction in predictions]",[np.argmax(prediction) for prediction in predictions])
        return [np.argmax(prediction) for prediction in predictions]


    def SaveModel(self, save_dir):
        model_json = self.model.to_json()
        with open(save_dir, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to:"+ str(self.model_dir))


    def LoadModel(self, load_dir):
        loaded_model_json = ""
        with open(load_dir, 'r') as f:
            loaded_model_json = f.read()
        
        loaded_model = model_from_json(loaded_model_json)
        print("Loaded model from:"+ str(self.model_dir))

    ### Model Specific Functions
    def BuildModel(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,dropout):
        model = Sequential()
        model.add(InputLayer(input_tensor=self.input_))
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',kernel_initializer=keras.initializers.glorot_uniform(),
                         input_shape=[model_input_dim_height, model_input_dim_width, model_input_channels], name="conv_1"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name="max_pool_1"))
        model.add(Conv2D(64, (3, 3), activation='relu',name="conv_2"))
        model.add(MaxPooling2D(pool_size=(2, 2),name="max_pool_2"))
        model.add(Flatten(name="feature_vector_1"))
        model.add(Dense(1048, activation='relu', name="fully_connected_1"))
        model.add(Dropout(dropout,name="dropout_1"))
        # model.add(Dense(n_classes, activation='relu',name="logits"))
        model.add(Dense(n_classes, activation='softmax',name="class_prob"))

        return model
        
    def GetWeights(self):
        return [w for w in self.model.weights if 'connected' in w.name and 'kernel' in w.name]
    
    def GetPlaceholders(self):
        return [self.input_, self.labels_]

    def GetGradLoss(self):
        return tf.gradients(self.loss, self.GetWeights())

    def GetLayerByName(self,name):
        print("GetLayerByName - not implemented")
    
    def FetchAllVariableValues(self):
        print("FetchAllVariableValues - not implemented")

    def _get_batches(self, x, y, batch_size):
        if x.shape[0] < batch_size:
            batch_size = x.shape[0]
        idcs = random.sample(range(x.shape[0]), batch_size)
        batch_x = x[idcs]
        batch_y = y[idcs]
        x = np.delete(x, idcs, 0)
        y = np.delete(y, idcs)
        return batch_x, batch_y, x, y

if __name__ == '__main__':

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    model_input_dim_height = 28
    model_input_dim_width = 28
    model_input_channels = 1
    n_classes = 10
    learning_rate = 0.001

    batch_size = 128
    num_train_steps = 200

    additional_args = {"learning_rate": learning_rate}

    cnn_model = KerasCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,
                          model_dir="mnist", additional_args=additional_args)

    verbose_every = 10
    for step in range(verbose_every, num_train_steps + 1, verbose_every):
        print("")
        print("training")
        print("step:", step)
        cnn_model.TrainModel(mnist.train.images, mnist.train.labels, batch_size, verbose_every)
        print("")

        print("evaluation")
        print(cnn_model.EvaluateModel(mnist.test.images[:128], mnist.test.labels[:128], batch_size))
        print("")

    print(cnn_model.Predict(mnist.test.images[:5]))

    print(mnist.test.labels[:5])
    
