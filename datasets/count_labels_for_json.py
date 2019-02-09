import os

def ProduceLabelCount(image_dir,dataset_folder):
	count_dict = {}

	dataset_folder_path = os.path.join(image_dir,dataset_folder)

	class_folders = os.listdir(dataset_folder_path)
	class_folders = [calss_folder for calss_folder in class_folders if calss_folder[0] != "."]
	class_folders.sort()


	for class_folder in class_folders:
		class_folder_path = os.path.join(dataset_folder_path,class_folder)

		images = os.listdir(class_folder_path)
		images = [img for img in images if img[0] != "."]
		
		count_dict[class_folder] = len(images)
	
	return count_dict


def LabelCountDictToJsonText(label_count_dict):
	label_count_text = ""

	labels = label_count_dict.keys()
	labels.sort()

	label_count_text += '{"label":"'+labels[0]+'","images":'+str(label_count_dict[labels[0]])+'}'
	for label in labels[1:]:
		label_count_text += ',\n\t\t\t\t{"label":"'+label+'","images":'+str(label_count_dict[label])+'}'

	return label_count_text

def GenerateLabelCountJson(image_dir,dataset_folder):
	return LabelCountDictToJsonText(ProduceLabelCount(image_dir,dataset_folder))

if __name__ == '__main__':
	image_dir = "dataset_images"
	dataset_folder = "imagenet_vehicles_birds_10_class_resized"

	label_count_json = GenerateLabelCountJson(image_dir,dataset_folder)
	print(label_count_json)
