import os
import urllib

host = "localhost"
port = "6101"

dataset_name = "gun_wielding_image_classification"
model_name = "cnn_1"

model_url = "http://"+host+":"+port+"/models/archive?dataset_name="+dataset_name+"&model_name="+model_name


output_path = os.path.join(model_name,"saved_models","zipped_models")


if(not os.path.exists(output_path)):
    os.mkdir(output_path)

output_path = os.path.join(output_path,dataset_name.replace(" ","_")+".zip")

urllib.urlretrieve(model_url, output_path)