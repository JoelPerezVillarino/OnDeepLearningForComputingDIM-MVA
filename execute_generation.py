import os
from src import DataGen


# json_path = os.path.join(os.getcwd(), "cir_data.json")
# json_path = os.path.join(os.getcwd(), "cir_data_2.json")
# json_path = os.path.join(os.getcwd(), "hull_white_data.json")
json_path = os.path.join(os.getcwd(), "json_files", "hull_white_data_portfolio.json")
gen = DataGen.from_json(json_path)
gen.gen_train_set()
gen.gen_val_set()

del gen

json_path = os.path.join(os.getcwd(), "json_files", "cir_data_portfolio.json")
gen = DataGen.from_json(json_path)
gen.gen_train_set()
gen.gen_val_set()

del gen

json_path = os.path.join(os.getcwd(), "json_files", "cir_data_extreme.json")
gen = DataGen.from_json(json_path)
gen.gen_val_set_adhoc(a=[0.5, 2, 7.5], sigma=[0.01, 0.05, 0.1])
