import os

observations = []


image_dir = "dataset_images"
dataset_folder = "wielder_non-wielder"
use_explicit = False

dataset_folder_path = os.path.join(image_dir,dataset_folder)

class_folders = os.listdir(dataset_folder_path)
class_folders = [calss_folder for calss_folder in class_folders if calss_folder[0] != "."]
class_folders.sort()


for class_folder in class_folders:
	class_folder_path = os.path.join(dataset_folder_path,class_folder)

	images = os.listdir(class_folder_path)
	images = [img for img in images if img[0] != "."]
	images.sort()


	for image in images:
		image_path = os.path.join(dataset_folder,class_folder,image)
		
		if(use_explicit):
			observations.append((str(os.path.abspath(image_path)),str(class_folder)))
		else:
			observations.append((str(image_path),str(class_folder)))


output_string = "image_path,label\n"

for observation in observations:
	output_string += observation[0] + "," + observation[1] + "\n"

with open(os.path.join("dataset_csvs",dataset_folder+".csv"),"w") as f:
	f.write(output_string[:-1])