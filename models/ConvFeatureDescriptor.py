import numpy as np
import os
import tensorflow as tf

from urllib.request import urlretrieve
from os.path import isdir, isfile
from tqdm import tqdm


if not isdir('tensorflow_vgg'):
    raise Exception("`tensorflow_vgg` is not found.\n \\ Have you cloned it? \\ (https://github.com/machrisaa/tensorflow-vgg)")

from tensorflow_vgg import vgg16, utils



class ConvFeatureDescriptor(object):
    """
        Uses https://github.com/machrisaa/tensorflow-vgg to produce a feature
        vector for a given image.
    """

    def __init__(self, batchsize, model=vgg16.Vgg16()):
        "load the architecture + corresponding weights"
        self.batchsize = batchsize

        # 1. Verify that 'tensorflow_vgg' is installed


        self.download_vggnet() if not isfile("tensorflow_vgg/vgg16.npy") else print("VGG parameters found.")

        self.model = model

        self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input_image")

        self.model = vgg16.Vgg16()

        self.model.build(self.input_)


    def get_feature_vectors_from_dir(self, datadir):
        " "
        contents = os.listdir(datadir)
        classes = [each for each in contents if os.path.isdir(datadir + each)]

        labels, batch = [], []
        codes = None

        with tf.Session() as sess:
            with tf.name_scope('content_vgg'):
                input_=tf.placeholder(tf.float32, [None,224,224,3])
                self.model.build(input_)

            for each in tqdm(classes):
                print("Starting {} images".format(each))
                class_path = datadir + each
                files = os.listdir(class_path)
                for ii, file in enumerate(files, 1):
                    img = utils.load_image(os.path.join(class_path, file))
                    batch.append(img.reshape((1, 224, 224, 3)))
                    labels.append(each)

                    if ii % self.batchsize == 0 or ii == len(files):
                        images = np.concatenate(batch)
                        codes_batch = self.convolve_features(sess, images)
                        if codes is None:
                            codes = codes_batch
                        else:
                            codes = np.concatenate((codes, codes_batch))

                    batch = []
                    print('{} images processed'.format(ii))

            return codes, labels

    def get_feature_vectors_from_images(self, images):
        codes = None
        with tf.Session() as sess:
            for i in range(0,images.shape[0]):
                img = images[i]
                codes_batch = self.convolve_features(sess, img)
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

            return codes

    def convolve_features(self, sess, img):
        feed_dict = {self.input_: img}
        codes_batch = sess.run(self.model.pool5, feed_dict=feed_dict)
        return codes_batch

    def download_vggnet(self):

        print("Downloading VGG16 Parameters...")
        with self.DLProgress(unit="B", unit_scale=True, miniters=1, desc="VGG16 params") as pbar:
            urlretrieve(
                "https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy",
                "tensorflow_vgg/vgg16.npy",
                pbar.hook,
            )

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

# USAGE
# d = ConvFeatureDescriptor()
# x = d.get_feature_vectors('data_dir') # one dir worth only for now
# x[0] => codes, x[1] => labels
