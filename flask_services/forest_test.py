import demo_path
from models.utils.ConvFeatureDescriptor import ConvFeatureDescriptor
from models.random_forest.random_forest import RandomForest

# descriptor = ConvFeatureDescriptor()
# TODO: sort out vgg16 to accept different dims
forest = RandomForest(x_dim=224, y_dim=224, n_channels=1, n_classes=2, model_dir="")

forest.InitialiseModel()