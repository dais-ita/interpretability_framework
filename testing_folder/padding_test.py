import numpy as np 

import cv2

image_batch = []

image_path = "00054_gun_wielder.jpg"

for i in range(4):
	image_batch.append(cv2.imread(image_path))

image_batch = np.array(image_batch)

image_array_shape = image_batch.shape

if(len(image_array_shape) == 4):
    image_shape = image_array_shape[1:]
else:
    image_shape = image_array_shape

min_height = 139
min_width = 139

target_shape = (max(min_height,image_shape[0]),max(min_width,image_shape[1]),image_shape[2])

shape_difference = (np.array(target_shape) - np.array(image_shape))

print(shape_difference)

cv2_image = image_batch[0]
cv2.imshow("image 0",cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

add_top = int(shape_difference[0]/2)
add_bottom = shape_difference[0] - add_top

add_left = int(shape_difference[1]/2)
add_right = shape_difference[1] - add_left

image_batch = np.pad(image_batch,((0,0),(add_top,add_bottom),(add_left,add_right),(0,0)), mode='constant', constant_values=0)

print(image_batch.shape)


cv2_image = image_batch[0]
cv2.imshow("image 0",cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
