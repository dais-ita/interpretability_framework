import csv
import math
import random
import numpy as np
import urllib
from PIL import Image
import os
from skimage import io
import cv2

class DataSet(object):
	"""docstring for DataSet"""
	def __init__(self, file_path, image_url_column, ground_truth_column):
		super(DataSet, self).__init__()
		self.file_path = file_path
		self.image_url_column = image_url_column
		self.ground_truth_column = ground_truth_column
		
		self.data = [] #all x-y pairs from the dataset
		self.ground_truth_labels = {} #the set of ground truth labels and how many examples of each are present in the data
		self.one_hot_list = [] #the order for one hot encoding of labels

		self.live_dataset = [] #the subset of data being used as the 'full dataset'
		
		self.live_training = []
		self.live_validation = []
		self.live_test = []		

	def ProduceData(self):
		data = []

		with open(self.file_path,"r") as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:
				data.append( (row[self.image_url_column],row[self.ground_truth_column]) )

		print("total data points loaded: "+str(len(data)))
		return data


	def CreateLiveDataSet(self, dataset_max_size = -1, even_examples=True, y_labels_to_use=[]):
		if len(self.data) == 0:
			self.data = self.ProduceData()
		
		live_data = self.FilterData(self.data,dataset_max_size,even_examples,y_labels_to_use)
		
		self.live_dataset = live_data

		return live_data

	def SplitLiveData(self,train_ratio=0.8,validation_ratio=0.1,test_ratio=0.1):
		
		total_examples = len(self.live_dataset)

		remaining_index_set = set(range(total_examples))

		num_train = int(math.floor(total_examples * train_ratio))
		num_validation = int(math.floor(total_examples * validation_ratio))
		num_test = total_examples - num_train - num_validation

		print("train: "+str(num_train)+"   validation: "+str(num_validation)+"  test: "+str(num_test))

		train_index = random.sample(remaining_index_set,num_train)
		remaining_index_set = remaining_index_set - set(train_index) 

		validation_index = random.sample(remaining_index_set,num_validation)
		remaining_index_set = remaining_index_set - set(validation_index) 

		full_source = np.array(self.live_dataset)

		self.live_training = full_source[train_index]
		self.live_validation = full_source[validation_index]
		self.live_test = full_source[list(remaining_index_set)]


	def GetBatch(self, batch_size = -1,even_examples=True, y_labels_to_use=[], split_batch = True,split_one_hot = True, batch_source = "full", return_batch_data=False):
		if len(self.live_dataset) == 0:
			self.live_dataset = self.CreateLiveDataSet(even_examples=even_examples, y_labels_to_use=y_labels_to_use)
			print("live dataset size: "+str(len(self.live_dataset)))

		if(batch_source == "full"):
			source = self.live_dataset

		elif(batch_source == "train"):
			source = self.live_training
		
		elif(batch_source == "validation"):
			source = self.live_validation
		
		elif(batch_source == "test"):
			source = self.live_test
		
		else:
			print("target batch source does not exist")
			return None

		batch = self.FilterData(source,batch_size,even_examples,y_labels_to_use)

		if(split_batch):
			if(return_batch_data):
				xs, ys = self.SplitDataXY(batch,split_one_hot)
				return xs, ys, batch
			else:
				return self.SplitDataXY(batch,split_one_hot)
		else:
			return batch


	def FilterData(self, input_data, max_return_size = -1, even_examples = True, y_labels_to_use = []):
		working_data = input_data

		if(len(working_data)) == 0:
			return working_data

		if(len(y_labels_to_use) > 0):
			working_data = self.FilterByLabels(working_data,y_labels_to_use)

		if max_return_size < 0:
			max_return_size = len(working_data)

		if(even_examples):
			live_gt_info = self.GetLabelInformation(working_data)

			print("gt_info:")
			print(live_gt_info)

			max_examples = min(live_gt_info.values())
			print("max_examples: "+str(max_examples))

			total_size_label_split = int(math.floor(float(max_return_size) / len(live_gt_info.keys())))

			num_examples = min(max_examples,total_size_label_split)

			split_data = {}

			for data_value in working_data:
				if(not data_value[1] in split_data):
					split_data[data_value[1]]=[]

				split_data[data_value[1]].append(data_value)

			working_data = []

			for label in split_data:
				working_data += random.sample(split_data[label], num_examples)

		else:
			working_data = random.sample(working_data, max_return_size)		

		return working_data	


	def FilterByLabels(self,data,labels_to_use):
		return [d for d in data if d[1] in labels_to_use]


	def GetLabelInformation(self,data):
		gt_info = {}

		for data_value in data:
				if(not data_value[1] in gt_info):
					gt_info[data_value[1]]=0

				gt_info[data_value[1]] += 1				
		
		return gt_info 		


	def SplitDataXY(self,data, one_hot_encoding = True,load_images = True):
		x_vals = []
		y_vals = []

		if(one_hot_encoding):
			if(len(self.one_hot_list) == 0):
				gt_info = self.GetLabelInformation(self.data)

				self.one_hot_list = list(gt_info.keys())
				self.one_hot_list.sort()

		labels = []
		for point in data:
			x_vals.append(point[0])

			if(one_hot_encoding):
				one_hot = [0]*len(self.one_hot_list)
				one_hot[self.one_hot_list.index(point[1])] = 1
				y_vals.append(one_hot)
				labels.append(point[1])
			else:
				y_vals.append(point[1])

	
		if(load_images):
			x_vals = self.LoadImagesfromURLs(x_vals)

		# self.SaveImages((x_vals*255).astype('uint8'),"test_images",labels)

		return x_vals, np.array(y_vals)


	def SaveLiveDataSet(self, csv_output_path, image_output_dir_path):

		if not os.path.exists(image_output_dir_path):
			os.mkdir(image_output_dir_path)

		x, y = self.SplitDataXY(self.live_dataset)

		y_labels = [val[1] for val in self.live_dataset]

		paths = self.SaveImages(x,image_output_dir_path,y_labels)

		self.SaveXY(paths,y,csv_output_path)


	def LoadImagesfromURLs(self,urls):
		images = []
		for url in urls:
			# print(url)
			images.append(self.ImageFromURL(url))
			# print(images[-1].shape)
		
		# print(images[-1])
		
		# img = Image.fromarray(images[0], 'RGB')
		# img.save('test_prearray.jpg')
		# img.show()
		
		images = np.array(images)

				
		# img = Image.fromarray(images[0], 'RGB')
		# img.save('test_array.jpg')
		# img.show()
		

		# print(images.shape)

		images = images.astype(np.float32)
		images = images / 255.0

		# print(images.shape)

		# print(images[0])
		
		# img = Image.fromarray(images[0], 'RGB')
		# img.save('test_float.jpg')
		# img.show()
		
		return images


	def ImageFromURL(self,url):
		try:
			return io.imread(url)
		except:
			return None


	def SaveImages(self, x_images, save_directory_path, y_labels = [],image_names=[]):
		paths = []
		for x_image_i in range(len(x_images)):
			if(len(image_names)):
				image_name = image_names[x_image_i]
			else:
				image_name = str(x_image_i)

			
			if(len(y_labels)):
				y_label = str(y_labels[x_image_i])
				label_dir = os.path.join(save_directory_path,y_label.replace(" ","_"))
				if(not os.path.exists(label_dir)):
					os.mkdir(label_dir)
				
				output_path = os.path.join(label_dir,image_name+"_"+y_label.replace(" ","_")+".jpg")
			
			else:
				output_path = os.path.join(save_directory_path,image_name+".jpg")

			# self.SaveNumpyAsJPG(output_path,x_images[x_image_i])
			paths.append(output_path)

		return paths


	def SaveNumpyAsJPG(self,path,image_array):
		im = Image.fromarray(image_array)
		im.save(path)


	def FilterImageURLS(self,urls,filter_size=(-1,-1,3)):
		filtered_urls = []
		filtered_images = []

		for url in urls:
			image_array = self.ImageFromURL(url)
			if(image_array is None):
				continue

			image_shape = image_array.shape
			print(image_shape)
			if(len(image_shape) != len(filter_size)):
				continue

			should_filter = False
			for dim_i in range(3):
				if(filter_size[dim_i] != -1 and filter_size[dim_i] != image_shape[dim_i]):
					should_filter = True
			if(should_filter):
				continue

			filtered_urls.append(url)
			filtered_images.append(image_array)
	
		return filtered_urls, np.array(filtered_images)

	
	def CreateFilteredCSV(self,output_path, save_images = True, image_dir = "dataset",batch_size=-1,filter_size=(-1,-1,3)):
		if len(self.data) == 0:
			self.data = self.ProduceData()
		
		self.CreateFilteredSet(self.data,output_path,image_dir,batch_size=batch_size,filter_size=filter_size)


	def CreateFilteredLiveCSV(self,output_path,dataset_max_size = -1, even_examples=True, y_labels_to_use=[], save_images = True,image_dir="dataset",batch_size=-1,filter_size=(-1,-1,3)):
		self.CreateLiveDataSet(dataset_max_size = -1, even_examples=True, y_labels_to_use=y_labels_to_use)

		self.CreateFilteredSet(self.live_dataset,output_path,image_dir,batch_size=batch_size,filter_size=filter_size)
		
	

	def CreateFilteredSet(self,dataset,output_path,image_dir,batch_size=-1,filter_size=(-1,-1,3)):
		if(not os.path.exists(image_dir)):
			os.mkdir(image_dir)

		x,y = self.SplitDataXY(dataset, one_hot_encoding = False,load_images = False)

		if(batch_size == -1):
			batch_size = len(x)

		num_batches = len(x) / batch_size
		last_batch_size = len(x) % batch_size
		if(last_batch_size):
			num_batches += 1

		print("number of batches: "+str(num_batches))
		for batch_num in range(num_batches):
			print("batch: "+str(batch_num))
			batch_start = batch_num * batch_size
			batch_end = min(len(x), (batch_num + 1) * batch_size)


			batch_x = x[batch_start:batch_end]
			batch_y = y[batch_start:batch_end]

			batch_x,batch_y = self.SkipPreFetched(batch_x,batch_y,image_dir=image_dir)

			if(len(batch_x) == 0):
				continue

			urls, images = self.FilterImageURLS(batch_x,filter_size=filter_size)

			image_names = [url.split("/")[-1].split("-")[0] for url in urls]

			paths = self.SaveImages(images,image_dir,batch_y,image_names)

			self.SaveXY(urls,batch_y,output_path,["image_url","label"],append_to_file = True)

			self.SaveXY(paths,batch_y,output_path.replace(".csv","_local.csv"),["image_path","label"],append_to_file = True)


	def SkipPreFetched(self,Xs,Ys,image_dir="dataset"):
		new_Xs = []
		new_Ys = []

		image_names = [url.split("/")[-1].split("-")[0] for url in Xs]


		for i in range(len(Xs)):
			folder_path = os.path.join(image_dir,Ys[i])
			image_path = os.path.join(folder_path,image_names[i]+"_"+Ys[i]+".jpg")
			if(os.path.exists(image_path)):
				continue
			new_Xs.append(Xs[i])
			new_Ys.append(Ys[i])

		return new_Xs,new_Ys


	def SaveXY(self,x,y,output_path,headings=[],append_to_file = False):
		output_string = ""

		if(len(headings)):
			if(not os.path.exists(output_path)):
				output_string += ",".join(headings) +"\n"

		if(type(y[0]) == list):
			for i in range(len(x)):
				output_string += x[i] + "," + "|".join(y[i]) + "\n"
		else:
			for i in range(len(x)):
				output_string += x[i] + "," + y[i] + "\n"

		if(append_to_file):
			with open(output_path,"a+") as csv_ouput_file:
				csv_ouput_file.write(output_string)
		else:
			with open(output_path,"w") as csv_ouput_file:
				csv_ouput_file.write(output_string)
		
	def ImageResize(self,image_dir,output_dir,target_width,target_height=-1,crop={'x1':-1,'x2':-1,'y1':-1,'y2':-1}):
		image_folders = os.listdir(image_dir)

		if(not os.path.exists(output_dir)):
			os.mkdir(output_dir)

		for folder in image_folders:
			folder_path = os.path.join(image_dir,folder)
			ims = os.listdir(folder_path)

			for image in ims[:]:
				image_path = os.path.join(folder_path,image)

				image_array = cv2.imread(image_path)

				
				crop_x1 = max(0,crop["x1"])
				crop_y1 = max(0,crop["y1"])

				if(crop["x2"] == -1):
					crop_x2 = image_array.shape[0]
				else:
					crop_x2 = crop["x2"]
				
				if(crop["y2"] == -1):
					crop_y2 = image_array.shape[1]
				else:
					crop_y2 = crop["y2"]

				croped = image_array[crop_x1:crop_x2,crop_y1:crop_y2,:]

				
				if(target_height == -1):
					target_height = int(math.floor(croped.shape[1]*(float(target_width)/croped.shape[0])))

				
				resized = cv2.resize(croped, (target_width,target_height), interpolation = cv2.INTER_AREA)

				output_folder = os.path.join(output_dir,folder)

				if(not os.path.exists(output_folder)):
					os.mkdir(output_folder)
			
				output_path = os.path.join(output_folder,image)

				cv2.imwrite(output_path,resized)


	def ConvertOneHotToClassNumber(self,y_array):
		return np.array([np.argmax(y) for y in y_array] ,np.int32)

if __name__ == '__main__':
	create_filtered_live_set = False


	if(create_filtered_live_set): #create and save a resized dataset if needed
		file_path = "wielder_non-wielder.csv"
		image_url_column = "image_path"
		ground_truth_column = "label"

		
		dataset_tool = DataSet(file_path,image_url_column,ground_truth_column)

		output_dir = "dataset"
		print("resizing")
		
		dataset_tool.CreateFilteredLiveCSV("live_dataset.csv",y_labels_to_use=["non_wielder","gun_wielder"],save_images = True,image_dir=output_dir,batch_size=30,filter_size=(300,300,3)) 



	file_path = "wielder_non-wielder.csv"
	image_url_column = "image_path"
	ground_truth_column = "label"

	
	dataset_tool = DataSet(file_path,image_url_column,ground_truth_column)
	x, y = dataset_tool.GetBatch(batch_size = 128,even_examples=True, y_labels_to_use=["non_wielder","gun_wielder"], split_batch = True,split_one_hot = True)

	print(len(x))
	print(x[0])
	print(x[0].shape)
	cv2_image = cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)

	cv2.imshow("image 0",cv2_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(y[0])

