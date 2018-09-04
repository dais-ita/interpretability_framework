import numpy as np

import os
import sys


def get_dir_path(dir_name):
    base_path = os.getcwd().split("/")
    while base_path[-1] != str(dir_name):
        base_path = base_path[:-1]
    return "/".join(base_path) + "/"


models_path = os.path.join(get_dir_path("p5_afm_2018_demo"), "models")
folders = os.listdir(models_path)
for folder in folders:
    folder_path = os.path.join(models_path, folder)
    if os.path.isdir(folder_path) and not folder_path in sys.path:
        sys.path.append(folder_path)

import keras
import tensorflow as tf

from keras.backend import set_session
from FeatureDescriptor import FeatureDescriptor

import random


class PremadeCNN(object):
    """A simple CNN model implemented using the Tensorflow estimator API"""

    def __init__(self, model_input_dim_width, model_input_dim_height, model_input_channels, n_classes, model_dir,
                 additional_args={}):
        super(PremadeCNN, self).__init__()
        self.model_input_dim_height = model_input_dim_height
        self.model_input_dim_width = model_input_dim_width
        self.model_input_channels = model_input_channels
        model_input_dim = [model_input_dim_height, model_input_dim_width, model_input_channels]
        self.input_ = tf.placeholder(
            tf.float32,
            shape=[None] + list(model_input_dim),
            name="in_"
        )
        self.labels_ = tf.placeholder(
            name="lbl_",
            shape=[None, 1],
            dtype=tf.float32
        )
        self.n_classes = n_classes
        self.model_dir = model_dir
        self.sess = tf.Session()
        self.feature_desc = None

        # model specific variables
        # Training Parameters

        if ("learning_rate" in additional_args):
            self.learning_rate = additional_args["learning_rate"]
        else:
            self.learning_rate = 0.001
            print("using default of " + str(self.learning_rate) + " for " + "learning_rate")
        #
        # if ("dropout" in additional_args):
        #     self.dropout = additional_args["dropout"]
        # else:
        #     self.dropout = 0.25
        #     print("using default of " + str(self.dropout) + " for " + "dropout")

        self.model = None
        if "architecture" in additional_args:
            self.architecture = additional_args["architecture"]
        else:
            self.architecture = "vgg16"
        self.InitaliseModel(model_dir=self.model_dir)

    ### Required Model Functions
    def InitaliseModel(self, model_dir="model_dir"):
        opts = tf.GPUOptions(allow_growth=True)
        conf = tf.ConfigProto(gpu_options=opts)
        # trainingConfig = tf.estimator.RunConfig(session_config=conf)
        self.sess = tf.Session(config=conf)
        keras.backend.set_session(self.sess.run)

        self.model = self.BuildModel(self.model_input_dim_height, self.model_input_dim_width, self.model_input_channels,
                                     self.n_classes)
        self.logits = self.model(self.input_)
        self.sess.run(tf.global_variables_initializer())
        self.loss = keras.losses.categorical_crossentropy(self.logits, self.labels_)

    def TrainModel(self, train_x, train_y, batch_size, num_steps, val_x=None, val_y=None):
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        while train_y.size != 0 and num_steps != 0:
            #     Get batch
            batch_x, batch_y, train_x, train_y = self._get_batches(train_x, train_y, batch_size)
            feed_dict = {
                self.input_: batch_x,
                self.labels_: batch_y
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

        loss = []
        num_steps = eval_x.shape[0] // batch_size
        while eval_y.size != 0 and num_steps != 0:
            batch_x, batch_y, eval_x, eval_y = self._get_batches(eval_y, eval_y, batch_size)
            feed_dict = {
                self.input_: batch_x,
                self.labels_: batch_y
            }
            loss.append(self.sess.run(self.loss, feed_dict))
            num_steps -= 1
        return sum(loss) / len(loss)

    def Predict(self, predict_x):
        if (type(predict_x) != dict):
            input_dict = {"input": predict_x}
        else:
            input_dict = predict_x

        feed_dict = {
            self.input_: predict_x
        }
        predictions = self.sess.run(self.logits, feed_dict)
        print("[np.argmax(prediction) for prediction in predictions]",
              [np.argmax(prediction) for prediction in predictions])
        return [np.argmax(prediction) for prediction in predictions]

    def SaveModel(self, save_dir):
        self.model.save_weights(os.path.join(save_dir, self.architecture + "weights.h5"))

    def LoadModel(self, load_dir):
        # TODO: get load model fns from Dan
        self.model.load_weights(os.path.join(save_dir, self.architecture + "weights.h5"))

    ### Model Specific Functions
    def BuildModel(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes):
        """
        Builds the internal model architecture by adapting the FeatureDescriptor class, with the simple alteration of
        setting the "include_top" parameter to true, to load the logits layers
        """
        featuredesc = FeatureDescriptor([model_input_dim_width, model_input_dim_height,model_input_channels], input_tensor=self.input_, architecture = self.architecture, include_top=True, weights='Random')
        model = featuredesc.get_premade_model()
        return model

    def GetWeights(self):
        return [w for w in self.model.trainable_weights if 'kernel' in w.name]

    def GetPlaceholders(self):
        return [self.input_, self.labels_]

    def GetGradLoss(self):
        return tf.gradients(self.loss, self.GetWeights())

    def GetLayerByName(self, name):
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



if __name__ == "__main__":
    n_classes = 2
    additional_args = {
        'learning_rate': 0.01,
        'alpha': 0.0001
    }
    batch_size = 10

    from PIL import Image as Pimage
    from tqdm import tqdm

    datadir = "/home/c1435690/Projects/DAIS-ITA/Development/p5_afm_2018_demo" + "/datasets/dataset_images/resized_wielder_non-wielder/"
    contents = os.listdir(datadir)
    classes = [each for each in contents if os.path.isdir(datadir + each)]

    labels, batch = [], []
    images = []

    for each in tqdm(classes):
        print("Starting {} images".format(each))
        class_path = datadir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            img = Pimage.open(os.path.join(class_path, file))
            img = np.array(img)
            batch.append(img)
            labels.append(each)

    images = np.asarray(batch)

    from sklearn import preprocessing

    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)

    labels_vecs = lb.transform(labels)
    labels_vecs[labels_vecs == 0] -= 1

    from sklearn.model_selection import StratifiedShuffleSplit

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    splitter = ss.split(np.zeros(labels_vecs.shape[0]), labels_vecs)

    train_idx, val_idx = next(splitter)

    half_val_len = int(len(val_idx) / 2)
    val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

    train_x, train_y = images[train_idx], labels_vecs[train_idx]
    val_x, val_y = images[val_idx], labels_vecs[val_idx]
    test_x, test_y = images[test_idx], labels_vecs[test_idx]

    input_dim = train_x[0].shape
    n_batches = 10

    svm_model = PremadeCNN(
        input_dim[0],
        input_dim[1],
        input_dim[2],
        n_classes,
        os.path.join(models_path, "premade_models"),
    )

    print("Train shapes (x,y):", train_x.shape, train_y.shape)
    print("Validation shapes (x,y):", val_x.shape, val_y.shape)
    print("Test shapes (x,y):", test_x.shape, test_y.shape)

    for step in range(10, 200 + 1, 10):
        print("")
        print("training")
        print("step:", step)
        svm_model.TrainModel(train_x, train_y, batch_size, 50)
        print("")
        print("evaluation")
        print(svm_model.EvaluateModel(val_x, val_y, batch_size))
        print("")

    test_y[test_y == 1] = 0
    test_y *= -1
    print(test_y == svm_model.Predict(test_x))
