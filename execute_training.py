import os
from src.training import Train


path_json = os.path.join(os.getcwd(),"cir_train.json")
train = Train.from_json(path_json)
train.run()