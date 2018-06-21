import numpy as np
from lime import lime_image

from skimage.color import gray2rgb, rgb2gray

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
        one_hot = [0] * self.model.n_classes
        one_hot[prediction] = 1
        one_hot_predictions.append(one_hot)

      return one_hot_predictions

  def MakeInputGray(self,X):
    return np.array([rgb2gray(img) for img in X],np.float32).reshape(X.shape[0], self.model.model_input_dim_width * self.model.model_input_dim_height)
    

  def ClassifyWithLIME(self,x,num_samples=1000,labels = (1,),top_labels=None,class_names = None):
    if(len(x.shape)== 4):
      print("classify function passed a 4d tensor, processing first element")
      x = x[0]

    print("x.shape",x.shape)
    x = x.reshape(self.model.model_input_dim_height,self.model.model_input_dim_height)
    print(x.shape)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(x, self.PredictFunction,labels=labels,top_labels=top_labels, hide_color=0,num_samples = num_samples)

    return self.PredictFunction(np.array([x])), explanation



if __name__ == '__main__':
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


  if(test_image.shape[-1] == 3 and self.model.model_input_channels == 1):
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