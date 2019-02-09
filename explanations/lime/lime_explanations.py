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
   # print("explanation - predictions: ",predictions)
    
    if(predictions[0].size == self.model.n_classes): #check if model prediction function returns one hot vector predictions
      return predictions
    else: #if it doesn't, convert predictions to one hot vectors
      #print("creating one-hot predections")
      one_hot_predictions = []
      for prediction in predictions:
        # print("explain prediciton", prediction)
        one_hot = [0] * self.model.n_classes
        one_hot[prediction] = 1
        one_hot_predictions.append(one_hot[:])
        # print("one_hot_predictions[-1]", one_hot_predictions[-1])
      #print("explanation - one_hot_predictions: ",one_hot_predictions)
      return one_hot_predictions

  def MakeInputGray(self,X):
    return np.array([rgb2gray(img) for img in X],np.float32).reshape(X.shape[0], self.model.model_input_dim_width * self.model.model_input_dim_height)



  def ClassifyWithLIME(self,x,num_samples=1000,labels = (1,),top_labels=None,class_names = None):
    if(len(x.shape)== 4):
      print("classify function passed a 4d tensor, processing first element")
      x = x[0]

    if(self.model.model_input_channels == 1 and x.shape[-1] == 3):
      print("input is 3 channel and model is single channel, converting to 1 channel")
      x = self.MakeInputGray(x)

    if(self.model.model_input_channels == 3 and x.shape[-1] == 1):
      print("input is single channel and model is 3 channel, converting to 3 channel")
      x = self.MakeInputRGB(x)

    
    x = x.reshape(self.model.model_input_dim_height,self.model.model_input_dim_height,self.model.model_input_channels)
    
    
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(x, self.PredictFunction,labels=labels,top_labels=top_labels, hide_color=0,num_samples = num_samples)


    # print("list(explanation.local_exp.keys())",list(explanation.local_exp.keys()))
    # print("")
    
    # print("explanation.local_exp[list(explanation.local_exp.keys())[0]].shape",explanation.local_exp[list(explanation.local_exp.keys())[0]])
    # print("")
    
    # print("explanation.local_exp[list(explanation.local_exp.keys())[0]].shape",explanation.local_exp[list(explanation.local_exp.keys())[1]])
    # print("")
    
    # print("explanation.segments.shape",explanation.segments.shape)
    # print("")
    
    # print("np.unique(explanation.segments)",np.unique(explanation.segments))
    # print("")
    
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

    prediction_scores, _ = self.model.Predict(np.array([input_image]), True)
    #print("lime prediction_scores",prediction_scores)
    prediction, explanation = self.ClassifyWithLIME(input_image,num_samples=num_samples,labels=list(range(self.model.n_classes)), top_labels=self.model.n_classes)
    
    #print("explanation prediction output",prediction)
    predicted_class = np.argmax(prediction)
    
    ## Calculate min_weight based on weights in this example:
    # print("")
    # print(explanation.local_exp[predicted_class])
    weights = [w[1] for w in explanation.local_exp[predicted_class]]
    largest_pro_evidence = max(weights)
    largest_against_evidence = min(weights)

    min_weight = min(largest_pro_evidence,abs(largest_against_evidence)) #set a minimum such that you will show the smallest number of regions that includes both the largest pro and largest con evidence. 

    # print(min_weight)

    #print("explanation predicted_class",predicted_class)
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=False, num_features=num_features, hide_rest=False,min_weight=min_weight)

    display_explanation_image = False

    if(display_explanation_image):
      cv2_image = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
      cv2.imshow("image 0",cv2_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    # explanation_image = mark_boundaries(temp,mask)

    explanation_image = temp
    explanation_image = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)

    if(not isinstance(prediction_scores,list)):
            prediction_scores = prediction_scores.tolist()

    additional_outputs = {"default_visualisation":temp.tolist(), "mask":mask.tolist(), "attribution_slices":explanation.segments.tolist(), "attribution_slice_weights":explanation.local_exp[predicted_class],"attribution_map":self.CreateAttributionMap(explanation.segments.tolist(),explanation.local_exp[predicted_class]).tolist(),"prediction_scores":prediction_scores[0]}

    explanation_text = "Evidence towards predicted class shown in green"
    return explanation_image, explanation_text, predicted_class, additional_outputs


  def CreateAttributionMap(self,attribution_slice,slice_weights):
    output_map = np.array(attribution_slice).astype(np.float32)

    for region_weight in slice_weights:
            # print(region_weight[0],region_weight[1])
      output_map[output_map == region_weight[0]] = region_weight[1]

    return output_map


if __name__ == '__main__':
  import os
  import sys
  

  ### Setup Sys path for easy imports
  # base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"
  # base_dir = "/media/upsi/fs1/harborned/repos/p5_afm_2018_demo"

  def GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo"):
    cwd = os.getcwd()
    split_cwd = cwd.split("/")

    base_path_list = []
    for i in range(1, len(split_cwd)):
      if(split_cwd[-i] == base_dir_name):
        base_path_list = split_cwd[:-i+1]

    if(base_path_list == []):
      raise IOError('base project path could not be constructed. Are you running within: '+base_dir_name)

    base_dir_path = "/".join(base_path_list)

    return base_dir_path

  base_dir = GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo")


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

  additional_args = {}

  cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir ="mnist", additional_args = additional_args )

  test_image = mnist.test.images[:1]
    
  lime_explainer = LimeExplainer(cnn_model)

  additional_args = {
  "num_samples":1000,
  "num_features":100,
  "min_weight":0.01
  }
  explanation_image, explanation_text, predicted_class, additional_outputs = lime_explainer.Explain(test_image,additional_args)
  
  # prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,labels=list(range(n_classes)),num_samples=10,top_labels=n_classes)
  # prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,num_samples=1000,labels=list(range(n_classes)), top_labels=n_classes)

  predicted_class = np.argmax(prediction)
  print("predicted_class",predicted_class)
  print("mnist.test.labels[:1]",mnist.test.labels[:1])

  print(explanation_text)
  cv2.imshow("explanation",explanation_image)


