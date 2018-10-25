import os

import base64
from PIL import Image
from StringIO import StringIO

import cv2 
import numpy as np

import urllib2

import json

from service_classes import DatasetServiceTool,ModelServiceTool,ExplanationServiceTool

def readb64(base64_string,convert_colour=True):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    img  = np.array(pimg)
    if(convert_colour):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)


def DisplayEncodedImage(encoded_string,convert_colour=True):
    bgr_img = readb64(encoded_string,convert_colour=convert_colour) #convert colour by default as image should be RGB and to display it needs to be BGR
    cv2.imshow("bgr_img (should display correctly)",bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    service_config = {
        "services":[
            {"service_type":"dataset",
                "ip":"localhost",
                "port":"6001"
            },
            {"service_type":"model",
                "ip":"localhost",
                "port":"6101"
            },
            {"service_type":"explanation",
                "ip":"localhost",
                "port":"6201"
            }
        ]
    }


    dataset_service_tool = DatasetServiceTool(service_config)

    datasets_jsons = dataset_service_tool.GetAvailableDatasets()

    dataset_names = [str(dataset_json["dataset_name"]) for dataset_json in datasets_jsons["datasets"]]
    dataset_names.sort()

    dataset_of_choice = dataset_names[1]
    selected_dataset_json = [dataset_json for dataset_json in datasets_jsons["datasets"] if dataset_json["dataset_name"] == dataset_of_choice][0]

    test_image_json = dataset_service_tool.GetTestImage(dataset_of_choice)

    print(test_image_json.keys())

    # DisplayEncodedImage(test_image_json["input"])


    model_service_tool = ModelServiceTool(service_config)

    available_models = model_service_tool.GetModelsForDataset(dataset_of_choice)
    available_models_names = [model["model_name"] for model in available_models["models"]]
    available_models_names.sort()

    print(available_models_names)

    model_of_choice = available_models_names[1]
    selected_model_json = [model_json for model_json in available_models["models"] if model_json["model_name"] == model_of_choice][0]


    # prediction_json = model_service_tool.RequestPrediction(test_image_json["input"],selected_dataset_json,model_of_choice)
    # print(prediction_json)


    explanation_service_tool = ExplanationServiceTool(service_config)
    
    available_explanations = explanation_service_tool.GetExplanationsForModelAndDataset(dataset_of_choice,model_of_choice)
    available_explanations_names = [explanation["explanation_name"] for explanation in available_explanations["explanations"]]
    available_explanations_names.sort()

    explanation_of_choice = "LRP"
    selected_explanation_json = [explanation_json for explanation_json in available_explanations["explanations"] if explanation_json["explanation_name"] == explanation_of_choice][0]



    print(available_explanations_names)

    
    explain_json = explanation_service_tool.ExplainPrediction(test_image_json["input"],selected_dataset_json,selected_model_json,selected_explanation_json)

    print(explain_json.keys())

    print("Prediction",explain_json["prediction"])
    DisplayEncodedImage(explain_json["explanation_image"])