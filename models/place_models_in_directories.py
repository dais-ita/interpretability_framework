import os
import shutil

trained_model_dir = "trained_model_files"

models_dir = "."

models_in_trained_dir = os.listdir(trained_model_dir)

for model in models_in_trained_dir:
	model_path = os.path.join(trained_model_dir,model)

	trained_model_files = os.listdir(model_path)

	for trained_model_file in trained_model_files:
		trained_model_path = os.path.join(model_path,trained_model_file)

		output_model_path = os.path.join(models_dir,model,"saved_models",trained_model_file)

		shutil.copy(trained_model_path,output_model_path)