import os
import keras
import cv2
import numpy as np
from PIL import Image

from multiprocessing import Pool

def KerasTransformImage(label,input_image,rotation_angle,shift_x,shift_y,shear,zoom_x,zoom_y,flip_horizontal,flip_vertical,channel_shift_intencity,brightness,generator=None):
    if(generator == None):
        generator = keras.preprocessing.image.ImageDataGenerator()
    transform_dict ={
        "theta":rotation_angle
        ,"tx":shift_x
        ,"ty":shift_y
        ,"shear":shear
        ,"zx":zoom_x
        ,"zy":zoom_y
        ,"flip_horizontal":flip_horizontal
        ,"flip_vertical":flip_vertical
        ,"channel_shift_intencity":channel_shift_intencity
        # ,"brightness":brightness
    }

    return (label,generator.apply_transform(input_image,transform_dict))

def p_helper(args):
    return KerasTransformImage(*args)

def ItterateKerasTransforms(input_image,rotation_angle_max,shift_x_max,shift_y_max,shear_max,zoom_x_max,zoom_y_max,flip_horizontal_max,flip_vertical_max,channel_shift_intencity_max,brightness_max,rotation_angle_min=0,shift_x_min=0,shift_y_min=0,shear_min=0,zoom_x_min=1,zoom_y_min=1,flip_horizontal_min=0,flip_vertical_min=0,channel_shift_intencity_min=0,brightness_min=0,rotation_angle_step=1,shift_x_step=1,shift_y_step=1,shear_step=5,zoom_x_step=0.1,zoom_y_step=0.1,flip_horizontal_step=1,flip_vertical_step=1,channel_shift_intencity_step=1,brightness_step=0.1,generator=None):
    p = Pool(20)

    new_images = []

    para_inputs=[]
    for rotation_angle in range(rotation_angle_min,rotation_angle_max+1,rotation_angle_step):
        for shift_x in range(shift_x_min,shift_x_max+1,shift_x_step):
            for shift_y in range(shift_y_min,shift_y_max+1,shift_y_step):
                for shear in range(shear_min,shear_max+1,shear_step):
                    for zoom_x in np.arange(zoom_x_min,zoom_x_max+(zoom_x_step/2),zoom_x_step):
                        zoom_x = round(zoom_x,2)
                        for zoom_y in np.arange(zoom_y_min,zoom_y_max+(zoom_y_step/2),zoom_y_step):
                            zoom_y = round(zoom_y,2)
                            for flip_horizontal in range(flip_horizontal_min,flip_horizontal_max+1,flip_horizontal_step):
                                for flip_vertical in range(flip_vertical_min,flip_vertical_max+1,flip_vertical_step):
                                    for channel_shift_intencity in range(channel_shift_intencity_min,channel_shift_intencity_max+1,channel_shift_intencity_step):
                                        # for brightness in np.arange(brightness_min,brightness_max+(brightness_step/2),brightness_step):
                                        brightness=0
                                        # print(label)
                                        label = (rotation_angle,shift_x,shift_y,shear,zoom_x,zoom_y,flip_horizontal,flip_vertical,channel_shift_intencity,brightness)
                                        para_inputs.append( [label,input_image,rotation_angle,shift_x,shift_y,shear,zoom_x,zoom_y,flip_horizontal,flip_vertical,channel_shift_intencity,brightness] )


    new_images =  p.map(p_helper, para_inputs)
    p.close()
    return new_images


def SaveImage(t_img,image_path,output_dir,root_dir_name):
    label_for_filename = str(t_img[0]).replace("(","").replace(")","").replace(".","-").replace(", ","_")
            
    relative_from_root_path = image_path.split(root_dir_name)[-1]
    
    output_path = os.path.join(output_dir,relative_from_root_path[1:]) 
    output_path = output_path.replace(".jpg","_"+label_for_filename + ".jpg")
    # if(not os.path.isdir(os.path.dirname(output_path))):
    #     os.makedirs(os.path.dirname(output_path))

    output_img = cv2.cvtColor(t_img[1], cv2.COLOR_BGR2RGB)
    im = Image.fromarray(output_img.astype(np.uint8))
    # print(output_path)
    im.save(output_path)

def save_helper(args):
    SaveImage(*args)

def CreateTransformedDataset(image_paths,output_dir,root_dir_name):
    rotation_angle_max = 45
    shift_x_max = 0
    shift_y_max = 0     
    shear_max = 10     
    zoom_x_max = 1.5    
    zoom_y_max = 1.5   
    flip_horizontal_max = 1     
    flip_vertical_max = 0 
    channel_shift_intencity_max = 0
    brightness_max = 0
    
    rotation_angle_step = 15
    zoom_x_step = 0.5
    zoom_y_step = 0.5

    p = Pool(20)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        new_images = ItterateKerasTransforms(img,rotation_angle_max,shift_x_max,shift_y_max,shear_max,zoom_x_max,zoom_y_max,flip_horizontal_max,flip_vertical_max,channel_shift_intencity_max,brightness_max,rotation_angle_step=rotation_angle_step,zoom_x_step=zoom_x_step,zoom_y_step=zoom_y_step)
        # print(len(new_images))
        save_args = []
        for t_img in new_images:
            save_args.append( [t_img,image_path,output_dir,root_dir_name] )
        
        p.map(save_helper, save_args)

    p.close()


if __name__ == '__main__':

    image_dir = "wielder_non-wielder"

    class_folders = os.listdir(image_dir)

    for class_folder in class_folders[1:]:
        print(class_folder)
        class_folder_path = os.path.join(image_dir,class_folder)

        class_images = os.listdir(class_folder_path)

        num_class_images = len(class_images)
        image_count = 0
        for img_name in class_images:
            # print(img_name,image_count,num_class_images)
            image_count += 1
            if(image_count%100==0):
                print(image_count)
            image_path = os.path.join(class_folder_path,img_name)

            output_dir="wielder_non-wielder_transformed"
            root_dir_name="wielder_non-wielder"
            CreateTransformedDataset([image_path],output_dir,root_dir_name)
