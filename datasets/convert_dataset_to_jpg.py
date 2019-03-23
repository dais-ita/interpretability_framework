import os
import sys

from PIL import Image

def ConvertImage(src_path,dst_path):
	im = Image.open(src_path)
	rgb_im = im.convert('RGB')
	rgb_im.save(dst_path)



if __name__ == '__main__':
	image_dir_path = "dataset_images"
	output_dir_path = "converted_images"
	if(not os.path.exists(output_dir_path)):
		os.mkdir(output_dir_path)

	for svrt_i in range(1,24):
		dataset_name = "svrt_problem_"+str(svrt_i)

		# if(len(sys.argv)>1):
		# 	dataset_name = os.argv[1]


		dataset_path = os.path.join(image_dir_path,dataset_name)
		output_dataset_path = os.path.join(output_dir_path,dataset_name)
		if(not os.path.exists(output_dataset_path)):
			os.mkdir(output_dataset_path)
		
		class_folders = os.listdir(dataset_path)

		for class_folder in class_folders:
			class_path = os.path.join(dataset_path,class_folder)
			output_class_path = os.path.join(output_dataset_path,class_folder)
			if(not os.path.exists(output_class_path)):
				os.mkdir(output_class_path)
		

			class_images = os.listdir(class_path)

			for class_image in class_images:
				class_image_path = os.path.join(class_path,class_image)
				output_class_image_path = os.path.join(output_class_path,class_image.replace(".png",".jpg"))

				ConvertImage(class_image_path,output_class_image_path)
				