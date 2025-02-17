import os

from src import DataGen


datafile = os.path.join(os.getcwd(), "cir_prueba.json")
generator = DataGen.from_json(datafile)
generator.gen_train_set()

generator.gen_val_set()