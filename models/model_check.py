import os
import shutil 

trained_model_dir = "trained_model_files"


model_dir = "."
model_folders = os.listdir(model_dir)

ignore_file_folders = ["trained_model_files", 'archived_depricated_models', 'utils', 'random_forest', 'model_check.py', 'download_models.py', 'models.json', 'keras_api_simple', 'conv_svm', 'svm']

model_folders = [folder for folder in model_folders if folder not in ignore_file_folders]


datasets = ["traffic_congestion_image_classification","traffic_congestion_image_classification_(resized)","gun_wielding_image_classification","cifar-10"]

models_available = {}

for model_folder in model_folders:
	print(model_folder)
	models_available[model_folder] = []
	saved_path = os.path.join(model_folder,"saved_models")

	if(os.path.exists(saved_path)):
		for dataset in datasets:
			print(dataset)
			dataset_model_path = os.path.join(saved_path,dataset+".h5")
			if(os.path.exists(dataset_model_path)):
				models_available[model_folder].append(dataset)

				
				#model
				output_model_path = os.path.join(trained_model_dir,model_folder)
				if(not os.path.exists(output_model_path)):
					os.mkdir(output_model_path)

				#dataset
				output_dataset_path = os.path.join(output_model_path,dataset+".h5")
				
				shutil.copy(dataset_model_path,output_dataset_path)


for key in models_available:
	print("model:"+key)
	for dataset in models_available[key]:
		print(dataset)
	print("")
	print("")
			