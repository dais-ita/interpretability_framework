import keras
import sys
import os
import json
import cv2
import numpy as np

def GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo"):
	cwd = os.getcwd()
	split_cwd = cwd.split("/")

	base_path_list = []
	for i in range(1, len(split_cwd)):
		if(split_cwd[-i] == base_dir_name):
			base_path_list = split_cwd[:-i+1]

	if(base_path_list == []):
		raise IOError('base project path could not be constructed. Are you running within: '+base_dir_name)

	base_dir_path = "/".join(base_path_list)

	return base_dir_path

base_dir = GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo")


datasets_path = os.path.join(base_dir,"datasets")
sys.path.append(datasets_path)

from DatasetClass import DataSet

dataset_name = "Gun Wielding Image Classification"

#### load dataset json
data_json_path = os.path.join(datasets_path,"datasets.json")

datasets_json = None
with open(data_json_path,"r") as f:
	datasets_json = json.load(f)


### get dataset details

dataset_json = [dataset for dataset in datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]


### gather required information about the dataset
file_path = dataset_json["ground_truth_csv_path"]
image_url_column = "image_path"
ground_truth_column = "label"
label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually
label_names.sort()
print(label_names)

input_image_height = dataset_json["image_y"]
input_image_width = dataset_json["image_x"]
input_image_channels = dataset_json["image_channels"]

### instantiate dataset tool
csv_path = os.path.join(datasets_path,"dataset_csvs",file_path)
dataset_images_dir_path =  os.path.join(datasets_path,"dataset_images")
dataset_tool = DataSet(csv_path,image_url_column,ground_truth_column,explicit_path_suffix =dataset_images_dir_path) #instantiates a dataset tool
dataset_tool.CreateLiveDataSet(dataset_max_size = -1, even_examples=True, y_labels_to_use=label_names) #creates an organised list of dataset observations, evenly split between labels

#load exisiting split
training_split_file = dataset_json["default_training_allocation_path"]
training_split_file_path = os.path.join(datasets_path,"dataset_csvs",training_split_file)
dataset_tool.ProduceDataFromTrainingSplitFile(training_split_file_path, explicit_path_suffix = dataset_images_dir_path)

source = "train"
train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)

train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    validation_split=0,
    fill_mode="constant"
)

train_data_gen.fit(train_x)

transform_param = {
    "theta": 90,
    "tx": 0.4,
    "ty": 0.4,
    "zx": 0.3,
    "zy": 0.3,
    "flip_horizontal": True,
    "flip_vertical": True,
    "shear": 0.3,
    "channel_shift_intencity": 0.0,
    "brightness": 0.0
}

aug_train_x = np.zeros_like(train_x)

batch_size = 1
i = 0

for x, y in train_data_gen.flow(train_x, train_y, batch_size):
    aug_train_x[i] = x
    i += 1

