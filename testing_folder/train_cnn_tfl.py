import sys
import os
import json

import cv2

import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off repeated messages from Tensorflow RE GPU allocation


### Setup Sys path for easy imports
# base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"
# base_dir = "/media/upsi/fs1/harborned/repos/p5_afm_2018_demo"

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



#add all model folders to sys path to allow for easy import
models_path = os.path.join(base_dir,"models")

model_folders = os.listdir(models_path)

for model_folder in model_folders:
	model_path = os.path.join(models_path,model_folder)
	sys.path.append(model_path)


#add all explanation folders to sys path to allow for easy import
explanations_path = os.path.join(base_dir,"explanations")

explanation_folders = os.listdir(explanations_path)

for explanation_folder in explanation_folders:
	explanation_path = os.path.join(explanations_path,explanation_folder)
	sys.path.append(explanation_path)


#add dataset folder to sys path to allow for easy import
datasets_path = os.path.join(base_dir,"datasets")
sys.path.append(datasets_path)
###


#import dataset tool
from DatasetClass import DataSet


#### load dataset json
data_json_path = os.path.join(datasets_path,"datasets.json")

datasets_json = None
with open(data_json_path,"r") as f:
	datasets_json = json.load(f)


### get dataset details
dataset_name = "Traffic Congestion Image Classification (Resized)"
dataset_json = [dataset for dataset in datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]


### gather required information about the dataset
file_path = dataset_json["ground_truth_csv_path"]
image_url_column = "image_path"
ground_truth_column = "label"
label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually
print(label_names)

input_image_height = dataset_json["image_y"]
input_image_width = dataset_json["image_x"]
input_image_channels = dataset_json["image_channels"]

### instantiate dataset tool
csv_path = os.path.join(datasets_path,"dataset_csvs",file_path)
dataset_images_dir_path =  os.path.join(datasets_path,"dataset_images")

dataset_tool = DataSet(csv_path,image_url_column,ground_truth_column,explicit_path_suffix =dataset_images_dir_path) #instantiates a dataset tool

dataset_tool.CreateLiveDataSet(dataset_max_size = -1, even_examples=True, y_labels_to_use=label_names) #creates an organised list of dataset observations, evenly split between labels
dataset_tool.SplitLiveData(train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1) #splits the live dataset examples in to train, validation and test sets


### get example batch and display an image
display_example_image = False

if(display_example_image):
	##select the source for the example
	# source = "full"
	# source = "train"
	# source = "validation"
	source = "test"

	x, y = dataset_tool.GetBatch(batch_size = 128,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)

	print(y[0])
	cv2_image = cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)
	cv2.imshow("image 0",cv2_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


### instantiate the model
model_json_path = os.path.join(models_path,"models.json")


models_json = None
with open(model_json_path,"r") as f:
	models_json = json.load(f)



model_json = models_json["models"][0]
print("selecting first model:" + model_json["model_name"])

print(model_json["script_name"]+"."+model_json["class_name"])
ModelModule = __import__(model_json["script_name"]) 
ModelClass = getattr(ModelModule, model_json["class_name"])


n_classes = len(label_names) 
learning_rate = 0.001

additional_args = {"learning_rate":learning_rate}

### load trained model

model_save_path = os.path.join(models_path,model_json["model_name"],dataset_name.lower().replace(" ","_"))
training_model = ModelClass(input_image_height, input_image_width, input_image_channels, n_classes, model_dir=model_save_path, additional_args=additional_args)


### train model
batch_size = 128
num_train_steps = 200

#load all train images as model handels batching
source = "train"
train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)

print("num train examples: "+str(len(train_x)))
train_y = dataset_tool.ConvertOneHotToClassNumber(train_y) #convert one hot vectors to class numbers as per model requirement

#validate on 128 images only
source = "validation"
val_x, val_y = dataset_tool.GetBatch(batch_size = 128,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
print("num validation examples: "+str(len(val_x)))
val_y = dataset_tool.ConvertOneHotToClassNumber(val_y) 


verbose_every = 10
for step in range(verbose_every,num_train_steps+1,verbose_every):
	print("")
	print("training")
	print("step:",step)
	training_model.TrainModel(train_x, train_y, batch_size, verbose_every)

	print("")
	print("evaluation")
	print(training_model.EvaluateModel(val_x, val_y, batch_size))
	print("")

training_model.SaveModel(model_save_path)


### test the model
source = "test"
test_x, test_y = dataset_tool.GetBatch(batch_size = 20,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
test_y = dataset_tool.ConvertOneHotToClassNumber(test_y) 

print("num test examples: "+str(len(test_x)))

print("predicted classes:")
print(training_model.Predict(test_x))


print("ground truth classes:")
print(test_y)



### use LIME to explain a classification
print("Generating LIME explanation")
from lime_explanations import LimeExplainer

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

lime_explainer = LimeExplainer(training_model)

test_image = test_x[0]
test_label = test_y[0]


explanation_image, explanation_text, prediction, additional_outputs = lime_explainer.Explain(test_image)

cv2_image = cv2.cvtColor(explanation_image, cv2.COLOR_RGB2BGR)
cv2.imshow("image 0",cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()