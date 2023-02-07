import json
import os
from copy import deepcopy


TEMP_JSON_PATH = "./fnc_exp.json"
with open(TEMP_JSON_PATH, 'r') as f:
    data = json.load(f)

data['DD']['pheno'] = '/home/yaser/yaser_temp/data/Phenotypic_V1_0b_preprocessed1.csv'
MAIN_DATA_FLDR = "/home/yaser/yaser_temp/ProcessedData/"

fldrs = [os.path.join(MAIN_DATA_FLDR, fldr) for fldr in os.listdir(MAIN_DATA_FLDR) if os.path.isdir(os.path.join(MAIN_DATA_FLDR, fldr)) ]
fldr_data_name = list(map(lambda x: x.split('/')[-1], fldrs))

abs_data_fldr_path_corr = list(map(lambda x: os.path.join("/home/yaser/yaser_temp/ProcessedData", x,"corr", "_CORR.csv"), fldr_data_name))
abs_data_fldr_path_dFC = list(map(lambda x: os.path.join("/home/yaser/yaser_temp/ProcessedData", x, "corr", "dFC.csv"), fldr_data_name))

for dest in abs_data_fldr_path_corr:
    filter_name = dest.split('/')[-3]
    d = deepcopy(data)
    d['DD']['data'] = dest
    with open(f"./ff_corr/{filter_name}.json",  'w') as f:
        json.dump(d, f)

for dest in abs_data_fldr_path_dFC:
    filter_name = dest.split('/')[-3]
    d = deepcopy(data)
    d['DD']['data'] = dest
    with open(f"./ff_dc/{filter_name}.json",  'w') as f:
        json.dump(d, f)