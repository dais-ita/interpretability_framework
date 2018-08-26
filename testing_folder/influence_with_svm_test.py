import os
from tqdm import tqdm
import numpy as np
import random

os.chdir("/home/c1435690/Projects/DAIS-ITA/Development/p5_afm_2018_demo")

from explanations.influence_functions.influence_explanations import InfluenceExplainer
from models.svm.ConvSVM import ConvSVM
from PIL import Image as Pimage




additional_args = {
    'learning_rate': 0.01,
    'alpha': 0.0001
}
batch_size = 70

datadir = os.getcwd() + "/datasets/dataset_images/resized_wielder_non-wielder/"
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

Pimage.fromarray(test_x[0])

input_dim = train_x[0].shape
n_batches = 10

print("Train shapes (x,y):", train_x.shape, train_y.shape)
print("Validation shapes (x,y):", val_x.shape, val_y.shape)
print("Test shapes (x,y):", test_x.shape, test_y.shape)

model = ConvSVM(input_dim[0], input_dim[1], input_dim[2],2, os.path.join(os.getcwd(), 'models/svm'),additional_args)
model.LoadModel()
influence_explainer = InfluenceExplainer(model, train_x, train_y, {})
rand_i = random.randint(0,test_y.shape[0] - 1)
additional_args = {'label': test_y[rand_i], 'collage': True}
test_img = Pimage.fromarray(test_x[rand_i])
test_img.save(os.path.join(os.getcwd(), 'testing_folder/test_image.png'))
expl = influence_explainer.Explain(test_x[rand_i], 10, additional_args)
expl.save(os.path.join(os.getcwd(), 'testing_folder/expl_images.png'))
