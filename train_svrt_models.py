from DaisFrameworkTool import DaisFrameworkTool

if __name__ == '__main__':#
	#ARGUMENTS
	model_name = "vgg16_imagenet"
	model_save_path_suffix = "" 

	dataset_name = "ImageNet Vehicles Birds 10 Class (Resized)"

	explanation_name = "LRP"

	if(len(sys.argv) > 1):
		dataset_name = sys.argv[1]
		
	if(len(sys.argv) > 2):
		model_name = sys.argv[2]
		

	if(len(sys.argv) > 3):
		model_save_path_suffix = sys.argv[3]
		

	if(len(sys.argv) > 4):
		explanation_name = sys.argv[4]
		
	print("Dataset Name: ", dataset_name)
	print("Model Name: ", model_name)
	print("Model Save Suffix: ", model_save_path_suffix)
	print("Explanation Name: ", explanation_name)


	#TRAINING PARAMETERS
	learning_rate = 0.000001
	batch_size = 128
	num_train_steps = 40
	train_model_after_loading = True

	#INSTANTIATE TOOL
	framework_tool = DaisFrameworkTool(explicit_framework_base_path="") ##if using the tool outside of the framework repo root folder, then you must provide an explicit path to it, otherwise use ""


	#LOAD DATASET
	dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name)

	label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually
	
	#LOAD TRAINING & VALIDATION DATA
	#load all train images as model handles batching
	print("load training data")
	print("")
	source = "train"
	train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
	
	print("num train examples: "+str(len(train_x)))


	#validate on 128 images only
	source = "validation"
	val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
	print("num validation examples: "+str(len(val_x)))

	

	
	#INSTANTIATE MODEL
	model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":learning_rate})
	
	#LOAD OR TRAIN MODEL
	load_base_model_if_exist = True
	train_model = train_model_after_loading
	
	#LOAD MODEL
	model_load_path = framework_tool.model_save_path
	if(load_base_model_if_exist == True and os.path.exists(model_load_path) == True):
		model_instance.LoadModel(model_load_path)
	else:
		train_model = True
	
	if(train_model):
		#OR TRAIN MODEL
		framework_tool.TrainModel(model_instance,train_x, train_y, batch_size, num_train_steps, val_x= val_x, val_y=val_y)




	#INSTANTIATE EXPLANTION
	if(explanation_name !=""):
			
		explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)

		additional_args = {
	        "num_samples":100,
	        "num_features":300,
	        "min_weight":0.01, 
	        "num_background_samples":50,
	        "train_x":train_x,
	        "train_y":train_y,
	        "max_n_influence_images":9,
	        "dataset_name":dataset_name,
	        "background_image_pool":train_x
	        }
		
		#EXPLAIN AN IMAGE
		image_x = train_x[0]

		explanation_image, explanation_text, predicted_class, additional_outputs = explanation_instance.Explain(image_x,additional_args=additional_args) 

		print("Prediction: ", predicted_class)

		cv2.imshow("Explanation Image",explanation_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()