import numpy as np
from lime import lime_image

from skimage.color import gray2rgb, rgb2gray

from skimage.segmentation import mark_boundaries

import cv2

class LimeExplainer(object):
  """docstring for LimeExplainer"""
  def __init__(self, model):
    super(LimeExplainer, self).__init__()
    self.model = model

  def PredictFunction(self,X):
    if(X.shape[-1] == 3 and self.model.model_input_channels == 1):
      X = self.MakeInputGray(X)
    
    predictions = list(self.model.Predict(X))

    if(predictions[0].size == self.model.n_classes): #check if model prediction function returns one hot vector predictions
      return predictions
    else: #if it doesn't, convert predictions to one hot vectors
      one_hot_predictions = []
      for prediction in predictions:
        # print("explain prediciton", prediction)
        one_hot = [0] * self.model.n_classes
        one_hot[prediction] = 1
        one_hot_predictions.append(one_hot[:])
        # print("one_hot_predictions[-1]", one_hot_predictions[-1])
        
      return one_hot_predictions

  def MakeInputGray(self,X):
    return np.array([rgb2gray(img) for img in X],np.float32).reshape(X.shape[0], self.model.model_input_dim_width * self.model.model_input_dim_height)
    

  def ClassifyWithLIME(self,x,num_samples=1000,labels = (1,),top_labels=None,class_names = None):
    # print("lime labels",labels)
    if(self.model.model_input_channels == 1 and x.shape[-1] == 3):
      X = self.MakeInputGray(test_image)

    if(len(x.shape)== 4):
      print("classify function passed a 4d tensor, processing first element")
      x = x[0]

    x = x.reshape(self.model.model_input_dim_height,self.model.model_input_dim_height,self.model.model_input_channels)
    
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(x, self.PredictFunction,labels=labels,top_labels=top_labels, hide_color=0,num_samples = num_samples)

    print("list(explanation.local_exp.keys())",list(explanation.local_exp.keys()))
    print("explanation.local_exp[list(explanation.local_exp.keys())[0]].shape",explanation.local_exp[list(explanation.local_exp.keys())[0]])
    print("explanation.segments.shape",explanation.segments.shape)
    return self.PredictFunction(np.array([x])), explanation


  def Explain(self,input_image, additional_args = {}):

    #load additional arguments or set to default
    if("num_samples" in additional_args):
      num_samples=additional_args["num_samples"]
    else:
      num_samples=30

    if("num_features" in additional_args):
      num_features=additional_args["num_features"]
    else:
      num_features=100

    if("min_weight" in additional_args):
      min_weight=additional_args["min_weight"]
    else:
      min_weight=0.01


    prediction, explanation = self.ClassifyWithLIME(input_image,num_samples=num_samples,labels=list(range(self.model.n_classes)), top_labels=self.model.n_classes)

    print("explanation prediction output",prediction)
    predicted_class = np.argmax(prediction)
    print("explanation predicted_class",predicted_class)
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=False, num_features=num_features, hide_rest=False,min_weight=min_weight)

    display_explanation_image = False

    if(display_explanation_image):
      cv2_image = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
      cv2.imshow("image 0",cv2_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    # explanation_image = mark_boundaries(temp,mask)

    explanation_image = temp

    additional_outputs = {"temp_image":temp, "mask":mask}

    explanation_text = "Evidence towards predicted class shown in green"
    return explanation_image, explanation_text, predicted_class, additional_outputs
    

if __name__ == '__main__':
  import os
  ### Setup Sys path for easy imports
  base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"
  base_dir = "/media/upsi/fs1/harborned/repos/p5_afm_2018_demo"
  #add all model folders to sys path to allow for easy import
  models_path = os.path.join(base_dir,"models")

  model_folders = os.listdir(models_path)

  for model_folder in model_folders:
    model_path = os.path.join(models_path,model_folder)
    sys.path.append(model_path)

  from CNN import SimpleCNN

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

  from skimage.segmentation import mark_boundaries
  import matplotlib.pyplot as plt

  model_input_dim_height = 28
  model_input_dim_width = 28 
  model_input_channels = 1
  n_classes = 10 


  cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes)

  test_image = mnist.test.images[:1]
    
  lime_explainer = LimeExplainer(cnn_model)


  if(test_image.shape[-1] == 3 and cnn_model.model_input_channels == 1):
      X = lime_explainer.MakeInputGray(test_image)

  # prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,labels=list(range(n_classes)),num_samples=10,top_labels=n_classes)
  prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,num_samples=1000,labels=list(range(n_classes)), top_labels=n_classes)

  predicted_class = np.argmax(prediction)
  print("predicted_class",predicted_class)
  print("mnist.test.labels[:1]",mnist.test.labels[:1])

  

  for i in range(n_classes):
    temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=100, hide_rest=False,min_weight=0.001)
    
    imgplot = plt.imshow(mark_boundaries(temp, mask))
    plt.show()