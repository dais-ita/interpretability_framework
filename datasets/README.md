# Datasets
Datasets are currently stored as part of the repo. In the future they will be moved to seperate storage and replaced in the repo by a download script. 

## JSON File
A json file is located in "datasets" that contains detatils of each dataset available. 
https://github.com/dais-ita/p5_afm_2018_demo/blob/master/datasets/datasets.json

## Dataset Format
Dataset images should be stored with the following structure:

- "datasets"
  - "dataset_images"
    - "[Dataset Name]" 
      - "[Class 1 Name]"
        - class 1, image 1
        - class 1, image 2
        - ...
        - class 1, image n

      - "[Class 2 Name]"
        - class 2, image 1
        - class 2, image 2
        - ...
        - class 2, image 
      
      - ...
       
      - "[Class N Name]"
        - class N, image 1
        - class N, image 2
        - ...
        - class N, image n

Images should be named "[unique_id]_[class_label].jpg" for easy class identification within applications.

A CSV that contains the dataset's image paths and label should be stored in "datasets/dataset_csvs"
columns: "image_path" and "label" respectively.
A python script is located in "datasets" that will create the csv given the presence of dataset structure correctly (as above).
Edit the script to point to the dataset folder and run. 



## Python Dataset Tool Class
A python class that makes loading, spliting and batching datasets easier. 
The class will work with datasets that have adhered to the above specification. 

Further documentation will be provided at a later date. The class script contains example code and a further example can be seen in the testing files location:
https://github.com/dais-ita/p5_afm_2018_demo/tree/master/testing_folder
