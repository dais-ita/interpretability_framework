import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout, Activation, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.models import model_from_json

from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class KerasApiVGG(object):
    """VGG-like CNN implemented with keras functional API"""

    def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir,
                 additional_args={}):
        super(KerasApiVGG, self).__init__()
        self.model_input_dim_height = model_input_dim_height
        self.model_input_dim_width = model_input_dim_width
        self.model_input_channels = model_input_channels
        self.n_classes = n_classes
        self.model_dir = model_dir

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
            self.dropout = 0.5
            print("using default of " + str(self.dropout) + " for " + "dropout")

        self.model = None
        self.InitaliseModel(model_dir=self.model_dir)

        self.sess = keras.backend.get_session()

        self.input_ = self.model.layers[0].input
        self.labels_ = tf.placeholder(tf.float32, shape = [None,])
        self.logits = self.model.layers[-1].output

        self.loss = keras.losses.categorical_crossentropy(self.labels_, self.logits)


    ### Required Model Functions
    def InitaliseModel(self, model_dir="model_dir"):
        opts = tf.GPUOptions(allow_growth=True)
        conf = tf.ConfigProto(gpu_options=opts)
        # trainingConfig = tf.estimator.RunConfig(session_config=conf)
        set_session(tf.Session(config=conf))

        self.model = self.BuildModel(self.model_input_dim_height, self.model_input_dim_width, self.model_input_channels, self.n_classes,self.dropout)
        

    def TrainModel(self, train_x, train_y, batch_size, num_steps, val_x= None, val_y=None):
        if (type(train_x) != dict):
            input_dict = {"input": train_x}
        else:
            input_dict = train_x

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=self.learning_rate),
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


    def EvaluateModel(self, eval_x, eval_y, batch_size):
        if (type(eval_x) != dict):
            input_dict = {"input": eval_x}
        else:
            input_dict = eval_x

        # Train the Model
        return self.model.evaluate(eval_x,eval_y, batch_size=batch_size)


    def Predict(self, predict_x):
        if (type(predict_x) != dict):
            input_dict = {"input": predict_x}
        else:
            input_dict = predict_x

        
        predictions = self.model.predict(predict_x)
        print("predictions:",predictions)
        print("[np.argmax(prediction) for prediction in predictions]",[np.argmax(prediction) for prediction in predictions])
        return [np.argmax(prediction) for prediction in predictions]


    def SaveModel(self, save_dir):
        model_json = self.model.to_json()
        with open(save_dir, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(save_dir+".h5")

        print("Saved model to:"+ str(save_dir+".h5"))


    def LoadModel(self, load_dir):
        if(load_dir[-3:] == ".h5"):
            load_h5_path = load_dir 
        else:
            load_h5_path = load_dir+".h5"

            self.model.load_weights(load_h5_path)
        
        print("Loaded model from:"+ str(load_h5_path))

    ### Model Specific Functions
    def BuildModel(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,dropout):
        vis_input = Input(shape=(model_input_dim_height, model_input_dim_width, model_input_channels), name="absolute_input")

        # x = ZeroPadding2D((1,1))                           (vis_input)
        # x = Convolution2D(32, 3, 3, activation='relu')     (x)
        
        x = Convolution2D(32, 3, 3, activation='relu')     (vis_input)
        
        x = Convolution2D(32, 3, 3, activation='relu')     (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = Convolution2D(64, 3, 3, activation='relu')    (x)
        x = Convolution2D(64, 3, 3, activation='relu')    (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = Flatten()                                      (x)
        x = Dense(256, activation='relu')                 (x)
        x = Dropout(dropout)                                   (x)
        x = Dense(n_classes, name="logits") (x)
        x = Activation('softmax',name="absolute_output") (x)
        model = Model(input=vis_input, output=x)
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

    cnn_model = KerasApiVGG(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,
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
