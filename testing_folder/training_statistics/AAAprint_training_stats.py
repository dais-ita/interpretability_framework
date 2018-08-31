import os

training_stats_dir_path = "."

training_stats_files = os.listdir(training_stats_dir_path)
training_stats_files = [f for f in training_stats_files if f[-3:] == "txt"]

training_stats_files.sort()


for training_file in training_stats_files:
	file_name_split = training_file.split("__")

	model_name = file_name_split[0]
	dataset_name = file_name_split[1]

	file_path = os.path.join(training_stats_dir_path,training_file)
	
	training_stats = None
	with open(file_path,"r") as f:
		stats_string = f.read()
		stats_lines = stats_string.split("\n")
		training_stats = stats_lines[1].split(",")

	print(model_name+","+dataset_name)
	print(training_stats)
	print("")