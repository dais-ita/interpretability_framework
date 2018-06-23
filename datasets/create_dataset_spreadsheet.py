import os

observations = []


image_dir = "dataset_images"
dataset_folder = "wielder_non-wielder"

dataset_folder_path = os.path.join(image_dir,dataset_folder)

class_folders = os.listdir(dataset_folder_path)

for class_folder in class_folders:
	class_folder_path = os.path.join(dataset_folder_path,class_folder)

	images = os.listdir(class_folder_path)

	for image in images:
		image_path = os.path.join(class_folder_path,image)

		observations.append((str(os.path.abspath(image_path)),str(class_folder)))


output_string = "image_path,label\n"

for observation in observations:
	output_string += observation[0] + "," + observation[1] + "\n"

with open(os.path.join("dataset_csvs",dataset_folder+".csv"),"w") as f:
	f.write(output_string[:-1])