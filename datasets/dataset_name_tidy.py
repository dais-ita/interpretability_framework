import os 

image_dir = "dataset_images"

dataset_name = "congested_non-congested"

dataset_image_path = os.path.join(image_dir,dataset_name)

class_labels = os.listdir(dataset_image_path)
class_labels.sort()

image_id_character_min = 5
image_id = -1

for class_label in class_labels:
	class_path = os.path.join(dataset_image_path,class_label)
	images = os.listdir(class_path)
	images.sort()

	for image in images:
		image_id += 1

		current_image_path = os.path.join(class_path,image)

		extension = "."+image.split(".")[-1]
		new_image_name = format(image_id,'05d')+"_"+class_label+extension 
		new_image_path = os.path.join(class_path,new_image_name)

		os.rename(current_image_path, new_image_path)