import os
from src.training import Train


path_json = os.path.join(os.getcwd(), "json_files","cir_train_2.json")
train = Train.from_json(path_json)
train.run()
