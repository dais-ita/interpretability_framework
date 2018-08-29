import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render

import cv2

model = models.InceptionV1()
model.load_graphdef()

print("starting visualisation")
images = render.render_vis(model, "mixed4a_pre_relu:476")

print(len(images))

print(images[0].shape)

display_explanation_image = True

if(display_explanation_image):
	cv2_image = images[0].reshape(images[0].shape[1],images[0].shape[2],images[0].shape[3])
	cv2.imshow("image 0",cv2_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()