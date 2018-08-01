import numpy as np 
import cv2

def Eval(array):
    return sum(sum(array))

def Operation(op_input,model):
    predictions = model.Predict(op_input)
    predictions.keys()
    return cnn_model.GetLayerByName("ConvNet/fully_connected_1/kernel")

def CreateInputs(n_inputs,x_shape,y_shape,channels):
    inputs = []

    for i in range(n_inputs):
        inputs.append(np.random.rand(x_shape,y_shape,channels).astype(np.float32))

    return inputs

def EvaluateInputs(inputs,model):
    evals = []
    for input_array in inputs:
        evals.append((input_array,Eval(Operation(input_array,model))))

    return evals

def EvolveInput(input_array,evolve_rate,mutation_chance):
    input_shape = input_array.shape
    evolution_up_pixels = np.random.rand(input_shape[0],input_shape[1],input_shape[2])
    evolution_up_pixels = evolution_up_pixels <= mutation_chance

    evolution_down_pixels = np.random.rand(input_shape[0],input_shape[1],input_shape[2])
    evolution_down_pixels = evolution_down_pixels <= mutation_chance

    evolution_mask = (evolution_up_pixels * evolve_rate) + (evolution_down_pixels * -evolve_rate)

    return input_array + evolution_mask


def EvolveStep(evaluations,evolve_rate,mutation_chance):
    new_generation = [input_array[0] for input_array in evaluations[:int(len(evaluations)/2)]]

    new_generation += new_generation[:]

    for i in range(len(new_generation)):
        new_generation[i] = EvolveInput(new_generation[i],evolve_rate,mutation_chance)

    return new_generation


if __name__ == '__main__':
    import sys
    import os
    import json

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off repeated messages from Tensorflow RE GPU allocation


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


    #add dataset folder to sys path to allow for easy import
    datasets_path = os.path.join(base_dir,"datasets")
    sys.path.append(datasets_path)
    ###

    #### load dataset json
    data_json_path = os.path.join(datasets_path,"datasets.json")

    datasets_json = None
    with open(data_json_path,"r") as f:
        datasets_json = json.load(f)


    ### get dataset details
    dataset_name = "Gun Wielding Image Classification"
    dataset_json = [dataset for dataset in datasets_json["datasets"] if dataset["dataset_name"] == dataset_name][0]


    ### gather required information about the dataset
    file_path = dataset_json["ground_truth_csv_path"]
    image_url_column = "image_path"
    ground_truth_column = "label"
    label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually
    label_names.sort()

    input_image_height = dataset_json["image_y"]
    input_image_width = dataset_json["image_x"]
    input_image_channels = dataset_json["image_channels"]



    ### instantiate the model
    model_json_path = os.path.join(models_path,"models.json")


    models_json = None
    with open(model_json_path,"r") as f:
        models_json = json.load(f)



    available_models = [model for model in models_json["models"] if dataset_name in [dataset["dataset_name"] for dataset in model["trained_on"]] ]
    print("available models:")
    print(available_models)
    print("")
    model_json = available_models[0]
    print("selecting first model:" + model_json["model_name"])

    print(model_json["script_name"]+"."+model_json["class_name"])
    ModelModule = __import__(model_json["script_name"]) 
    ModelClass = getattr(ModelModule, model_json["class_name"])


    n_classes = len(label_names) 
    learning_rate = 0.001

    additional_args = {"learning_rate":learning_rate}

    ### load trained model
    trained_on_json = [dataset for dataset in model["trained_on"] if dataset["dataset_name"] == dataset_name][0]

    model_load_path = os.path.join(models_path,"cnn_1",trained_on_json["model_path"])
    cnn_model = ModelClass(input_image_height, input_image_width, input_image_channels, n_classes, model_dir=model_load_path, additional_args=additional_args)
    cnn_model.LoadModel(model_load_path) ## for this model, this call is redundant. For other models this may be necessary. 


    x_dim = 128
    y_dim = 128
    channels = 3

    evolve_rate = 0.01
    mutation_chance = 0.01

    evolve_steps = 1000

    population_size = 10


    input_arrays = CreateInputs(population_size,x_dim,y_dim,channels)

    for evolve_step in range(evolve_steps):
        if(evolve_step%10 == 0):
            print("evolve_step",evolve_step)

        # print("len(input_arrays)",len(input_arrays))
        evaluations = EvaluateInputs(input_arrays,cnn_model)
        evaluations.sort(key=lambda x: x[1],reverse=True)
        print("fittest input:",evaluations[0][1]," weakest input:",evaluations[-1][1])
        input_arrays = EvolveStep(evaluations,evolve_rate,mutation_chance)


    cv2.imshow("best img",evaluations[0])

