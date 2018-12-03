# DAIS ITA - Interpretabity Framework
An interpretability technique exploration and data generation tool

This project provides a framework designed, from the ground up to allow easy building of dataset -> model -> explanation technqiue pipelines.

## Getting Started
steps marked with * only need to be done once
### Preparing Python* 
- install python *
- install requirements from requirements.txt *

### Adding/Downloading Datasets and Models 
#### Downloading Datasets*
- Download Archives from: https://drive.google.com/drive/folders/1tMYRvVOQkvJqyIV6XWjITEJgkOGCWxYf?usp=sharing
- unzip and place dataset root folder (e.g. "cifar-10" in datasets/dataset_images)
- repository should already have the required dataset csvs for the downloaded datasets but if you encounter errors (or you add your own datasets) edit and run the "create_dataset_spreadsheet.py" file
- repository should already have a datasets.json entry for the downloaded datasets but if you encounter errors (or if you add your own datasets) be sure the appropriate entries are found in datasets/datasets.json

#### Download Model Archive*
- Download the trained model archive from: https://drive.google.com/open?id=1x8_2kGe-0h0Fy_ECjpGvq5IzyRDqqo1Q
- extract "trained_model_files" folder to models/
- run models/place_models_in_directories.py to automatically take the models found in the extracted folder in to the correct locations within the models. 

#### Adding your own Datasets
(further documentation found: https://github.com/dais-ita/p5_afm_2018_demo/blob/master/datasets)
- place a root folder in datasets/dataset_images which contains sub-directories for each class label (root_folder/class_1 contains all the images from class 1 from your dataset)
- edit and run the "create_dataset_spreadsheet.py" file
- add an entry to datasets/datasets.json

#### Adding your own Models
(this is a top level description and deeper documentation will be provided in the future)
- wrap your model using the class structure defined for the framework (this is not yet documented but the existing models provide examples of how to do this)
- train your model and place the saved parameter file (or any files needed to load your trained model in the "saved_models" directory (How you load your model is determined by you in the "LoadModel" class function ).
- add an entry for your model to the models/models.json file


### Starting Python Services
- navigate to flask_services

run:

- python DatasetsAAS.py
- python ModelsAAS.py
- python ExplanationsAAS.py
- python DatastoreAAS.py

### Preparing and Starting Unification Layer
- Install npm *
- navigate to nodeJS/unification 
- run npm install *
- run npm start

### Open an interface
The framework is designed to take advantage of custom built interfaces that utilise the unification layer api, however it also includes general purpose interfaces. Using one such interface can be done by:
- navigate to: interfaces/interpretability_ui/scripts
- open interpretability_framework.js and ensure the variable api_base_url is set to the address of the unification api you wish to use (usually: "http:localhost:3100" but can be a remotely hosted api if desired)
- navigate to: interfaces/interpretability_ui
- open index.html in your browser of choice. 

## Framework Structure

The framework is structured in three layers:
  - Item and service layer
  - Unification API layer
  - Interface Layer
  
## Item and service layer
### Framework Items
The codebase for framework items (datasets, machine learning models and explanation techniques) are wrapped in classes designed to unify their input/output signature. This allows the framework to easily form pipelines from the different items and also allows for new items to be added to the framework easily. 

Instructions on how to add items to the framework can be found within the README of each item-type parent folder:
https://github.com/dais-ita/p5_afm_2018_demo/blob/master/datasets

https://github.com/dais-ita/p5_afm_2018_demo/blob/master/models

https://github.com/dais-ita/p5_afm_2018_demo/blob/master/explanations

### Item Services
For each item category, python flask services exist, located in https://github.com/dais-ita/p5_afm_2018_demo/blob/master/flask_services
These services handle the instantiation of the framework items and also handle the passing of data to and from the items. 

The services are not currently designed to handle multiple requests concurrently. This is due to the resource dependant nature of machine learning models and explanations. In future revisions, this capability can be added or a more distributed approach to the services can be implemented. The tool is designed for exploration by researchers as well as experiment designing and thus the current use cases do not require a single running instance of the framework to serve multiple users simultaneously. 


## Unification API layer
The unification API offers a single point to perform RESTful requests across the framework. The API service itself also offers a simple UI to explore the items within the framework and for many use cases, this UI may suffice. 

## Interface Layer
At the interface layer, custom interfaces can be built that take advantage of the unification API. Do to the REST API design, many different applications or interfaces can be built to take advantage of the simplistic pipeline building the API provides. 

## Available Items

For each option group, a JSON file exists that describes the existing options. The interface is built to understand available options and valid configurations via the JSON file. 

Available Datasets (https://github.com/dais-ita/p5_afm_2018_demo/blob/master/datasets)
- gun wielder image binary classification
- mnist
- cifar10
- TFL Traffic Congestion
- TFL Traffic Congestion (Resized)


Available Models
- VGG16
- VGG19
- Inception V3
- Inception ResNet V2
- Xception
- MobileNet


Available Explanations
- LIME
- LRP Deep Taylor
- Influence Functions
- Shapley


Available Adversarial Attack Generators
CleverHans (not yet integrated)
GenAttack (not yet integrated)
