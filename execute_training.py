import os
from src.training import Train


path_json = os.path.join(os.getcwd(), "json_files","hull_white_portfolio_train.json")
train = Train.from_json(path_json)
train.run(mode="single")
