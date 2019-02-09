import os

from count_labels_for_json import GenerateLabelCountJson

def GetSampleForInterestingImages(image_dir,dataset_folder):
	dataset_path = os.path.join(image_dir,dataset_folder)

	class_folders = os.listdir(dataset_path)

	interesting_images = []

	for class_folder in class_folders[:4]:
		class_path = os.path.join(dataset_path,class_folder)
		images = os.listdir(class_path)

		interesting_images.append(images[0])

	return interesting_images


image_dir = "dataset_images"

dataset_name = "ImageNet Vehicles Birds 10 Class (Resized)"
dataset_folder = "imagenet_vehicles_birds_10_class_resized"

labels_json = GenerateLabelCountJson(image_dir,dataset_folder)

image_x = "128"
image_y = "128"
image_channels = "3"

interesting_images_string = str(GetSampleForInterestingImages(image_dir,dataset_folder)).replace("'",'"')

description = ""

license = "CHECK IMAGENET"

dataset_json = '''{
	"dataset_name":"'''+dataset_name+'''",
	"download_link":"",
	"folder":"'''+dataset_folder+'''",
	"labels":[
				'''+labels_json+'''
			],
	"image_x":'''+image_x+''',
	"image_y":'''+image_y+''',
	"image_channels":'''+image_channels+''',
	"ground_truth_csv_path":"'''+dataset_folder+'''.csv",
	"default_training_allocation_path":"'''+dataset_folder+'''_split.csv",
	"interesting_images":'''+interesting_images_string+''',
	"references":[""],
	"description":"'''+description+'''",
	"license":"'''+license+'''"
}'''

print(dataset_json)