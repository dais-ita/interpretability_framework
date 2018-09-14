# DAIS ITA - Interpretabity Framework
A technique exploration and data generation tool

This project provides a framework designed, from the ground up, to allow easy building of dataset-ML model-explanation pipelines.

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


##Unification API layer
The unification API offers a single point to perform RESTful requests across the framework. The API service itself also offers a simple UI to explore the items within the framework and for many use cases, this UI may suffice. 

##Interface Layer
At the interface layer, custom interfaces can be built that take advantage of the unification API. Do to the REST API design, many different applications or interfaces can be built to take advantage of the simplistic pipeline building the API provides. 



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
