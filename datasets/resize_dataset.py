import os

from DatasetClass import DataSet

dataset_name = "mtg_set"

file_path = os.path.join("dataset_csvs","mtg_set.csv")
image_url_column = "image_path"
ground_truth_column = "label"

dataset_tool = DataSet(file_path,image_url_column,ground_truth_column)

image_dir = os.path.join("dataset_images",dataset_name)
output_dir = os.path.join("dataset_images",dataset_name+"_resized")


width = 352 #y
height = 288 #x 

## calculate crop around centre
mid_x = height / 2
mid_y = width / 2

target_crop_width = 200
target_crop_height = 200

x1 = mid_x - (target_crop_width/2)
x2 = mid_x + (target_crop_width/2)

y1 = mid_y - (target_crop_height/2)
y2 = mid_y + (target_crop_height/2)

#shift off centre as required 
shift_x = 0
shift_y = 0

x1 += shift_x
x2 += shift_x

y1 += shift_y
y2 += shift_y

#not resizing and just croping so target_width == taget_crop_width
target_width = 128
target_height = 128


dataset_tool.ImageResize(image_dir,output_dir,target_width,target_height=target_height,crop={'x1':x1,'x2':x2,'y1':y1,'y2':y2})