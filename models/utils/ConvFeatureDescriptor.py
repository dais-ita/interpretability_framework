import numpy as np
import os
import tensorflow as tf

from urllib.request import urlretrieve
from os.path import isfile
from os import mkdir
from tqdm import tqdm


try:
    from models.utils.tensorflow_vgg import vgg16, utils
except ImportError as error:
    print("`tensorflow_vgg not found.` Please ensure submodule \\"
          "https://github.com/machrisaa/tensorflow-vgg is installed.")


class ConvFeatureDescriptor(object):
    """
        Uses https://github.com/machrisaa/tensorflow-vgg to produce a feature
        vector for a given image.
        USAGE:
        descriptor = ConvFeatureDescriptor(verbose=0, mode='array', x_dim=28, y_dim=28, channels=1)
        X = descriptor(array_of_images)
        X # array of feature vectors
    """

    def __init__(self, verbose=True, mode='FILE', batch_size=10, x_dim=224, y_dim=224, channels=3):
        """load the architecture + corresponding weights"""
        self.batch_size = batch_size
        self.verbose = verbose
        self.mode = str(mode).lower()  # 'FILE' or 'IMAGE'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.channels = channels

        file_path = os.path.abspath(__file__).rsplit('/',1)[0]

        self.__download_vggnet() if not isfile(str(file_path) + "/tensorflow_vgg/vgg16.npy") else print("VGG parameters found.")
        self.model = vgg16.Vgg16()
        self.input = tf.placeholder(tf.float32, [None, self.x_dim, self.y_dim, self.channels], name="input_image")

        self.model.build(self.input)

    def get_feature_vectors(self, data):

        if self.mode == 'file':
            return self.__features_from_file(data)
        if self.mode == 'images':
            return self.__features_from_images(data)

    def __features_from_images(self, data):
        """
            img: array of images
        """
        codes = None
        with tf.Session() as sess:
            # test
            codes = self.__process_features(sess, data)

            # for i in range(0, data.shape[0]):
            #     # TODO: figure out a placeholder so that sess.run is only run once per call.
            #     codes_batch = self.__process_features(sess, data[i])
            #     if codes is None:
            #         codes = codes_batch
            #     else:
            #         codes = np.concatenate((codes, codes_batch))

        return codes

    def __process_features(self, sess, img):
        feed_dict = {self.input: img}
        codes_batch = sess.run(self.model.pool5, feed_dict=feed_dict)

        return codes_batch

    def __features_from_file(self, datadir):

        contents = os.listdir(datadir)
        classes = [each for each in contents if os.path.isdir(datadir + each)]

        # TODO: Check if data already exists, else create data folder and start routine.

        codes_list, labels, batch = [], [], []
        codes = None

        with tf.Session() as sess:
            with tf.name_scope('content_vgg'):
                self.model.build(self.input)

            for each in tqdm(classes):
                print("Starting {} images".format(each))
                class_path = datadir + each
                files = os.listdir(class_path)
                for ii, file in enumerate(files, 1):
                    img = utils.load_image(os.path.join(class_path, file))
                    batch.append(img.reshape((1, self.x_dim, self.y_dim, self.channels)))
                    labels.append(each)

                    if ii % self.batch_size == 0 or ii == len(files):
                        images = np.concatenate(batch)
                        codes_batch = self.__process_features(sess, images)

                        if codes is None:
                            codes = codes_batch
                        else:
                            codes = np.concatenate((codes, codes_batch))

                    batch = []
                    print('{} images processed'.format(ii))

            return codes, labels

    def __download_vggnet(self):

        print("Downloading VGG16 Parameters...")
        with DLProgress(unit="B", unit_scale=True, miniters=1, desc="VGG16 params") as progress:
            urlretrieve(
                "https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy",
                "./utils/model_utils/tensorflow_vgg/vgg16.npy",
                progress.hook
            )


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num



# #TODO
# class ImageLoader(object):
#     ''' '''
#     def __init__(self, json_path=None, data_dir=None):
#         self.json_path = json_path
#         self.data_dir  = data_dir
#
#     def get_dataset_description(self, json_path):
#         # Parse the JSON path and print data while returning paths as array
#
#     def get_images(self, data_dir):
#         return
#
#     def get_labels(self, data_dir):
#         return

