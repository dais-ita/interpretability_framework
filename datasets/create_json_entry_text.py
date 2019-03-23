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


def CreateJsonEntryText(dataset_attributes,image_dir = "dataset_images"):
	
	dataset_name = dataset_attributes["dataset_name"]
	dataset_folder = dataset_attributes["dataset_folder"]

	image_x = dataset_attributes["image_x"]
	image_y = dataset_attributes["image_y"]
	image_channels = dataset_attributes["image_channels"]

	description = dataset_attributes["description"]
	license = dataset_attributes["license"]

	labels_json = GenerateLabelCountJson(image_dir,dataset_folder)

	

	interesting_images_string = str(GetSampleForInterestingImages(image_dir,dataset_folder)).replace("'",'"')

	

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
		
	return dataset_json


if __name__ == '__main__':
	# dataset_attributes = {
	# 	"dataset_name": "ImageNet Vehicles Birds 10 Class (Resized)"
	# 	,"dataset_folder": "imagenet_vehicles_birds_10_class_resized"

	# 	,"image_x": "128"
	# 	,"image_y": "128"
	# 	,"image_channels": "3"

	# 	,"description": ""
	# 	,"license": ""
	# }
		

	# dataset_json_text = CreateJsonEntryText()
	# print("")
	# print("")
	# print(dataset_json_text)
	# print("")
	# print("")	

	dataset_jsons = []


	for i in range(1,24):
		dataset_attributes = {
		"dataset_name": "SVRT Problem "+str(i)
		,"dataset_folder": "svrt_problem_"+str(i)

		,"image_x": "128"
		,"image_y": "128"
		,"image_channels": "3"

		,"description":"Examples from problem "+str(i)+" of the SVRT. Images containing shapes are either generated following a specific rule (positive) or they explicitly don't follow the rule (negative)"
		,"license":"apache 2"
		}

		dataset_jsons.append(CreateJsonEntryText(dataset_attributes))


	print(",\n\t".join(dataset_jsons))

	
