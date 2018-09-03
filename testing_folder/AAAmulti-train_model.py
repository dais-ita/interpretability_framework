import sys
import os
import json

import cv2

import numpy as np
import random

import time

######
## Training COnfiguration ##

#DATASET
# dataset_name = "Traffic Congestion Image Classification"
# dataset_name = "Traffic Congestion Image Classification (Resized)"
# dataset_name = "Gun Wielding Image Classification"
# dataset_name = "CIFAR-10"

dataset_names = ["Traffic Congestion Image Classification",
"Traffic Congestion Image Classification (Resized)",
"Gun Wielding Image Classification",
"CIFAR-10"]

dataset_names = ["Traffic Congestion Image Classification"]


# dataset_names = ["Gun Wielding Image Classification"]

#MODEL
# model_name = "conv_svm"
# model_name = "keras_api_vgg"
# model_name = "keras_api_simple"
# model_name = "vgg16_imagenet"
# model_name = "vgg19_imagenet"
# model_name = "inception_v3_imagenet"
# model_name = "inception_resnet_v2_imagenet"
# model_name = "mobilenet_imagenet"
# model_name = "xception_imagenet"

model_names = ["vgg16_imagenet",
"vgg19_imagenet", 
"inception_v3_imagenet",
"inception_resnet_v2_imagenet",
"mobilenet_imagenet",
"xception_imagenet"]


model_names = [
"inception_resnet_v2_imagenet",
"mobilenet_imagenet",
"xception_imagenet"]


model_names = [
"inception_v3_imagenet"]


# model_names = ["inception_v3_imagenet",
# "inception_resnet_v2_imagenet",
# "mobilenet_imagenet",
# "xception_imagenet"]

# model_names = ["mobilenet_imagenet"]

#TRAINING PARAMETERS
learning_rate = 0.0000001
batch_size = 64
num_train_steps = 1000


#####


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

### load model json
model_json_path = os.path.join(models_path,"models.json")

models_json = None
with open(model_json_path,"r") as f:
	models_json = json.load(f)



	for dataset_name in dataset_names:

		### get dataset details
		dataset_json = [dataset for dataset in datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]


		### gather required information about the dataset
		if("default_training_allocation_path" in dataset_json.keys()):
			file_path = dataset_json["default_training_allocation_path"]
			load_split = True
		else:
			file_path = dataset_json["ground_truth_csv_path"]
			load_split = False
			print("new training split will be created")


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

		print("label_names",label_names)
		dataset_tool.CreateLiveDataSet(dataset_max_size = -1, even_examples=True, y_labels_to_use=label_names) #creates an organised list of dataset observations, evenly split between labels

		if(load_split):
			dataset_tool.ProduceDataFromTrainingSplitFile(csv_path,explicit_path_suffix =dataset_images_dir_path)
		else:
			dataset_tool.SplitLiveData(train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1) #splits the live dataset examples in to train, validation and test sets

		for model_name in model_names:
			### instantiate the model
			print("instantiating model")
			
			model_json = [model for model in models_json["models"] if model["model_name"] == model_name ][0]
			print("selecting first model:" + model_json["model_name"])

			print(model_json["script_name"]+"."+model_json["class_name"])
			ModelModule = __import__(model_json["script_name"]) 
			ModelClass = getattr(ModelModule, model_json["class_name"])

			n_classes = len(label_names) 

			additional_args = {"learning_rate":learning_rate}

			### load trained model
			model_save_path = os.path.join(models_path,model_json["model_name"],"saved_models","test_"+dataset_name.lower().replace(" ","_"))
			model_save_path = os.path.join(models_path,model_json["model_name"],"saved_models",dataset_name.lower().replace(" ","_"))
			model_instance = ModelClass(input_image_height, input_image_width, input_image_channels, n_classes, model_dir=model_save_path, additional_args=additional_args)

			if(os.path.exists(model_save_path)):
				model_instance.LoadModel(model_save_path)



			### train model

			#load all train images as model handels batching
			print("load training data")
			print("")
			source = "train"
			train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
			random.shuffle(train_x)

			print("num train examples: "+str(len(train_x)))


			#validate on 128 images only
			source = "validation"
			val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
			print("num validation examples: "+str(len(val_x)))


			if(model_name == "cnn_1"):
				train_y = dataset_tool.ConvertOneHotToClassNumber(train_y) #convert one hot vectors to class numbers as per model requirement
				val_y = dataset_tool.ConvertOneHotToClassNumber(val_y) 
				model_validates_during_test = False

			else:
				model_validates_during_test = True

			print("train model")
			print("")
			model_train_start_time = time.time()
			if(model_validates_during_test):
				model_instance.TrainModel(train_x, train_y, batch_size, num_train_steps, val_x= val_x, val_y=val_y)
			else:
				verbose_every = 10
				for step in range(verbose_every,num_train_steps+1,verbose_every):
					print("")
					print("training")
					print("step:",step)
					model_instance.TrainModel(train_x, train_y, batch_size, verbose_every)

					print("")
					print("evaluation")
					print(model_instance.EvaluateModel(val_x, val_y, batch_size))
					print("")


			print("saving model")
			model_instance.SaveModel(model_save_path)
			model_train_time = time.time() - model_train_start_time
			accuracy_after_training = model_instance.EvaluateModel(val_x, val_y, batch_size)
			print(model_train_time)
			print(accuracy_after_training)

			completed_epochs = str(len(model_instance.model.history.history["loss"]))

			train_stats_dir = "training_statistics"
			output_file_name = model_name+"__"+dataset_name+"__"+str(num_train_steps)+".txt"

			output_path = os.path.join(train_stats_dir,output_file_name)

			if(isinstance(accuracy_after_training,list)):
				output_string = "steps,learning_rate,batch_size,time,loss,accuracy\n"+str(completed_epochs)+","+str(learning_rate)+","+str(batch_size)+","+str(model_train_time)+","+str(accuracy_after_training[0])+","+str(accuracy_after_training[1])
			else:
				output_string = "steps,learning_rate,batch_size,time,accuracy\n"+str(completed_epochs)+","+str(learning_rate)+","+str(batch_size)+","+str(model_train_time)+","+str(accuracy_after_training)

			with open(output_path,"w") as f:
				f.write(output_string)

		# ### test the model
		# print("load test data")
		# print("")
		# source = "test"
		# test_x, test_y = dataset_tool.GetBatch(batch_size = 20,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
		# test_y = dataset_tool.ConvertOneHotToClassNumber(test_y) 

		# print("num test examples: "+str(len(test_x)))

		# print("predicted classes:")
		# print(cnn_model.Predict(test_x))


		# print("ground truth classes:")
		# print(test_y)


