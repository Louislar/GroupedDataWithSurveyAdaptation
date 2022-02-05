'''
確認精簡過的code(並且使用config.py作為參數輸入)的輸出與舊有code的輸出是否相同
'''

import pandas as pd 
from os import listdir
from os.path import join, isfile

num_of_year = 3
# Data Generated
first_dir = './simulation_data_gen_testing/simul_data/'
second_dir = './config_version/simul_data/'
# preprocessed data
first_dir = './simulation_data_gen_testing/simul_data/matrix_and_vector/'
second_dir = './config_version/simul_data/matrix_and_vector/'
# Data for QP
# first_dir = './simulation_data_gen_testing/simul_data/qp_input_output/'
# second_dir = './config_version/simul_data/qp_input_output/'

# Only check files not directories
first_files_list = [join(first_dir, f) for f in listdir(first_dir) if isfile(join(first_dir, f))]
second_files_list = [join(second_dir, f) for f in listdir(second_dir) if isfile(join(second_dir, f))]
first_files_list = sorted(filter(lambda x: isfile(join(first_dir, x)), listdir(first_dir)))
second_files_list = sorted(filter(lambda x: isfile(join(second_dir, x)), listdir(second_dir)))
print(first_files_list)
print(second_files_list)


# for f in listdir(first_dir): 
#     if isfile(join(first_dir, f)): 
#         print('file: ', join(first_dir, f))
#     else: 
#         print('dir: ', join(first_dir, f))

for _first_file, _second_file in zip(first_files_list, second_files_list): 
    print(_first_file, ', ', _second_file)
    _first_file = join(first_dir, _first_file)
    _second_file = join(second_dir, _second_file)
    tmp_first = pd.read_csv(_first_file)
    tmp_second = pd.read_csv(_second_file)
    print(tmp_first.equals(tmp_second))

# for i in range(num_of_year): 
#     tmp_first = pd.read_csv(first_dir+'year_{0}.csv'.format(i))
#     tmp_second = pd.read_csv(second_dir+'year_{0}.csv'.format(i))
#     isSame = tmp_first.equals(tmp_second)
#     print(isSame)