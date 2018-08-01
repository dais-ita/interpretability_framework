import numpy as np
import os
import tensorflow as tf

from urllib.request import urlretrieve
from os.path import isfile, isdir
from os import mkdir
from tqdm import tqdm

try:
    from utils.tensorflow_vgg import vgg16, utils
except ImportError as error:
    print("`tensorflow_vgg not found.` Please ensure submodule \\"
          "https://github.com/machrisaa/tensorflow-vgg is installed.")


class ConvFeatureDescriptor(object):
    """
        Uses https://github.com/machrisaa/tensorflow-vgg to produce a feature
        vector for a given image.
        USAGE:
        descriptor = ConvFeatureDescriptor(x_dim=28, y_dim=28, channels=1)
        X = descriptor(array_of_images)
        X # array of feature vectors
        'FILE' mode transforms a dataset from a directory into an equal size directory
        'IMAGES' mode transforms a dataset from an input argument to a return value
    """

    def __init__(self, batch_size=50, x_dim=224, y_dim=224, channels=3):
        """load the architecture + corresponding weights"""
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.channels = channels
        script_directory = os.path.abspath(__file__).rsplit('/', 1)[0]  # todo - add this to PATH so that import works
        self.vgg_path = str(script_directory) + '/tensorflow_vgg/vgg16.npy'

        self.__download_vggnet() if not isfile(self.vgg_path) else print("Found VGG16 Parameters. Building Model...")

        self.input_ = tf.placeholder(tf.float32, [None, self.x_dim, self.y_dim, self.channels], name="input_image")
        model = vgg16.Vgg16()

        with tf.name_scope('content_vgg'):
            model.build(self.input_)

        self.model = model

    def get_feature_vectors(self, images):
        """ Computes feature vectors from arg `images` TODO: batch it so that sess isn't inefficient """
        codes = None
        with tf.Session() as sess:
            for i in range(0, images.shape[0]):
                img = images[i]
                codes_batch = self.__process_features(sess, img)
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

        return codes

    def __process_features(self, sess, img):
        feed_dict = {self.input_: img}
        codes_batch = sess.run(self.model.fc6, feed_dict=feed_dict)
        return codes_batch

    def __download_vggnet(self):

        print("Downloading VGG16 Parameters...")
        with DLProgress(unit="B", unit_scale=True, miniters=1, desc="VGG16 params") as progress:
            urlretrieve(
                "https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy",
                self.vgg_path,
                progress.hook
            )


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num



if __name__ == "__main__":
    d = ConvFeatureDescriptor()

    (X, y) = d.get_feature_vectors(
        "/Users/c1524413/Projects/p5_afm_2018_demo/datasets/dataset_images/wielder_non-wielder/")

    X