import numpy as np
from lime import lime_image

class LimeExplainer(object):
  """docstring for LimeExplainer"""
  def __init__(self, model):
    super(LimeExplainer, self).__init__()
    self.model = model

  def PredictFunction(self,X):
    print(X.shape)

    predictions = list(self.model.Predict(X))

    print(len(predictions))
    print(predictions)
    print("")
    if(predictions[0].size == self.model.n_classes): #check if model prediction function returns one hot vector predictions
      return predictions
    else: #if it doesn't, convert predictions to one hot vectors
      one_hot_predictions = []
      for prediction in predictions:
        one_hot = [0] * self.model.n_classes
        one_hot[prediction] = 1
        one_hot_predictions.append(one_hot)

      print(one_hot_predictions)  
      return one_hot_predictions

    

  def ClassifyWithLIME(self,x,num_samples=1000,labels = (1,),top_labels=None,class_names = None):
    if(len(x.shape)== 4):
      print("classify function passed a 4d tensor, processing first element")
      x = x[0]

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(x, self.PredictFunction,labels=labels,top_labels=top_labels, hide_color=0,num_samples = num_samples)

    return self.predict(np.array([x])), explanation



if __name__ == '__main__':
  from CNN import SimpleCNN

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

  model_input_dim_height = 28
  model_input_dim_width = 28 
  model_input_channels = 1
  n_classes = 10 


  cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes)

  print(cnn_model.Predict(mnist.test.images[:5]))

  print(mnist.test.images[:5].shape)

  lime_explainer = LimeExplainer(cnn_model)

  explanation = lime_explainer.ClassifyWithLIME(mnist.test.images[:1],num_samples=10,labels=list(range(10)),top_labels=1)