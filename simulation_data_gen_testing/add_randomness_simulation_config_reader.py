'''
將參數儲存成json格式，並且儲存在每一次實驗當中
使用這個code讀取參數的.json檔案
'''

import json
from types import SimpleNamespace 
# from simulation_with_random_switch.lognormal_no_noise_random_cohort import config
import importlib

main_directory = './simulation_with_random_switch/lognormal_no_noise_random_cohort/'

def parse_dir_file_to_import_format(dir_path): 
    return dir_path[2:-1].replace('/', '.') + '.config'


# string import 
# ref: https://stackoverflow.com/questions/8718885/import-module-from-string-variable
config = importlib.import_module(parse_dir_file_to_import_format(main_directory))

print(config.main_directory)

if __name__ == '__main__': 
    print(parse_dir_file_to_import_format(main_directory))


