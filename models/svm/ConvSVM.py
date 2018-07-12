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

import numpy as np
import tensorflow as tf
import random
import ConvFeatureDescriptor

class ConvSVM(object):
    """A linear SVM using features extracted from the convolutional layers of a pre-trained NN"""

    def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, model_dir, addit_args, n_classes=2):
        super(ConvSVM, self).__init__()
        self.model_input_dim_height = model_input_dim_height
        self.model_input_dim_width = model_input_dim_width
        self.model_input_channels = model_input_channels
        if model_input_dim_height != 224 or model_input_dim_width != 224 or model_input_channels != 3:
            raise Exception("The input dimensions cannot be used with VGG")
        self.n_classes = n_classes
        self.checkpoint_path = os.path.join(model_dir,"checkpoints")

        self.sess = tf.Session()
        

        if "learning_rate" in addit_args:
            self.learning_rate = addit_args["learning_rate"]
        else:
            print("No learning rate specified in additional_args\nUsing default value of 0.001")
            self.learning_rate = 0.001
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate)

        if "alpha" in addit_args:
            self.alpha = addit_args["alpha"]


        self.feature_desc = ConvFeatureDescriptor.ConvFeatureDescriptor(mode='images')

        self.input_ = None
        self.labels_ = None

        self.initialised = False

        self.saver = None


    def InitialiseModel(self, n_features):
        self.input_ = tf.placeholder(
            name="in_",
            shape=[None, n_features],
            dtype=tf.float32
        )
        self.labels_ = tf.placeholder(
            name="lbl_",
            shape=[None, 1],
            dtype=tf.float32
        )

        self.W = tf.Variable(
            tf.random_normal(
                shape=[n_features, 1]
            ),
            name="W"
        )
        self.b = tf.Variable(
            tf.random_normal(
                shape=[1, 1]
            ),
            name="b"
        )




    def Output(self):
        mdl_out = tf.subtract(
            tf.matmul(
                self.input_,
                self.W
            ),
            self.b
        )
        return mdl_out


    def GetPredictions(self):
        return tf.sign(self.Output())


    def Loss(self):
        l2_norm = tf.reduce_sum(self.W)
        classif_term = tf.reduce_mean(
            tf.maximum(
                0.,
                tf.subtract(
                    1.,
                    tf.multiply(
                        self.Output(),
                        self.labels_
                    )
                )
            )
        )
        loss = tf.add(
            classif_term,
            tf.multiply(
                self.alpha,
                l2_norm
            )
        )
        return loss

    def get_batches(self, x, y, batch_size):
        idcs = random.sample(range(x.shape[0]),batch_size)
        return x[idcs], y[idcs]


    def TrainModel(self, train_x, train_y, batch_size, n_steps):
        train_x = self.feature_desc.get_feature_vectors(train_x, batch=True)
        train_x = train_x.reshape(len(train_y), -1)
        if not self.initialised:
            self.InitialiseModel(train_x.shape[1])
            self.saver = tf.train.Saver()
            self.initialised = True

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_steps):
                x, y = self.get_batches(train_x, train_y, batch_size)
                sess.run(
                    self.optim.minimize(self.Loss()),
                    feed_dict={
                        self.input_: x,
                        self.labels_: y
                    }
                )
            self.SaveModel(sess)


    def EvaluateModel(self, val_x, val_y, batch_size):
        val_x = self.feature_desc.get_feature_vectors(val_x,batch=True)
        val_x = val_x.reshape(len(val_y), -1)

        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    self.GetPredictions(),
                    self.labels_
                ),
                tf.float32
            )
        )

        with tf.Session() as sess:
            self.LoadModel(sess)
            accuracies = []
            for i in range(val_y.shape[0]//batch_size):
                x, y = self.get_batches(val_x, val_y, batch_size)
                accuracies.append(
                    sess.run(
                        acc,
                        feed_dict={
                            self.input_: x,
                            self.labels_: y
                        }
                    )
                )
            return tf.reduce_mean(accuracies).eval()

    def Predict(self, predict_x):
        if len(predict_x.shape) < 4:

            predict_x = predict_x.reshape(1, 224, 224, 3)
            batch = False
        else:
            batch = True
        n_samples = predict_x.shape[0]
        
        predict_x = self.feature_desc.get_feature_vectors(predict_x,batch=batch)
        predict_x = predict_x.reshape(n_samples, -1)

        if not self.initialised:
            self.InitialiseModel(predict_x.shape[1])
            self.saver = tf.train.Saver()
            self.initialised = True

        predictions = self.GetPredictions()
        with tf.Session() as sess:
            self.LoadModel(sess)
            predictions = sess.run(
                predictions,
                feed_dict={
                    self.input_: predict_x
                }
            )
            predictions[predictions == 1] = 0
            predictions *= -1
            one_hot = []
            for y in predictions:
                if y == 0:
                     one_hot.append([1,0])
                else:
                     one_hot.append([0,1])   
            return np.asarray(one_hot)

    def SaveModel(self,sess):
        self.saver.save(sess, os.path.join(self.checkpoint_path,"wvnw_svm.ckpt"))

    def LoadModel(self,sess):
        self.saver.restore(sess, os.path.join(self.checkpoint_path,"wvnw_svm.ckpt"))



if __name__ == '__main__':
    dim_height = 224
    dim_width = 224
    input_channels = 3
    n_classes = 2
    additional_args = {
        'learning_rate': 0.01,
        'alpha': 0.0001
    }
    batch_size = 70

    svm_model = ConvSVM(
        dim_height,
        dim_width,
        input_channels,
        os.path.join(models_path,"svm"),
        additional_args,
        n_classes,
    )
    from tensorflow_vgg import vgg16, utils as vgg_utils
    from tqdm import tqdm


    datadir = "../../datasets/dataset_images/resized_wielder_non-wielder/"
    contents = os.listdir(datadir)
    classes = [each for each in contents if os.path.isdir(datadir + each)]

    labels, batch = [], []
    images = []

    for each in tqdm(classes):
        print("Starting {} images".format(each))
        class_path = datadir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            img = vgg_utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
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


    n_batches = 10

    print("Train shapes (x,y):", train_x.shape, train_y.shape)
    print("Validation shapes (x,y):", val_x.shape, val_y.shape)
    print("Test shapes (x,y):", test_x.shape, test_y.shape)


    for step in range(10, 200+1, 10):
        print("")
        print("training")
        print("step:", step)
        svm_model.TrainModel(train_x, train_y, batch_size, 50)
        print("")
        print("evaluation")
        print(svm_model.EvaluateModel(val_x, val_y, batch_size))
        print("")

    test_y[test_y==1]=0
    test_y *= -1
    print(test_y == svm_model.Predict(test_x))
