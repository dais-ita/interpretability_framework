# p5_afm_2018_demo
Project 5 AFM 2018 Demo - Modular Interpretability Interface

The demo presents an interface for exploring different explanation types of machine learning models.

- The user selects a task (by selecting a dataset and model to use).
- A list of available explanation types is generated for that task and presented. 
- The user selects the deseried explanation types and sets any parameters.
- The user runs a test image from the dataset through the selected model and is presented with an output and explanations for each type selected.

For some tasks, an adversarial attack generator is available. This generator will craft input examples from test data that appear unchanged to the human eye but will cause an incorrect class to be assigned by the model.
For models that offer the avility to test adversarial attacks, the select is made available during the task selection step.  


For each option group,a json file exists that describes the existing options. The interface is built to understand available options and valid configurations via the JSON file. 

Available Datasets
- gun wielder image binary classification
- mnist (not yet integrated)
- cifar10 (not yet integrated)
- gun vs not gun , airport airscan (https://www.kaggle.com/c/passenger-screening-algorithm-challenge/data ) (not yet integrated) 


Available Models
- CNN 1 (with model structure A) (not yet integrated)
- CNN 2 (With model structure B) (not yet integrated)
- Pretrained CNN (not yet integrated)
- Pretrained CNN with transfer learning (not yet integrated)
- SVM (not yet integrated)
- RandomForest (not yet integrated)


Available Explanations
- LIME (not yet integrated)
- LRP Deep Taylor (not yet integrated)
- Salient Semantic Objects (not yet integrated)
- Influence Functions (not yet integrated)
- Training Data Examples (not yet integrated)
- surrogate model - decision tree (not yet integrated)


Available Adversarial Attack Generators
CleverHans (not yet integrated)
GenAttack (not yet integrated)
