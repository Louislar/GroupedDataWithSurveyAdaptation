'''
Preprocess the simulation data generate by generate_simul_data.py
- Compute probability vectors and population vectors
- Compute transition matrices 

(
    1. simulation資料中的資訊抽取and 整理，儲存給後面分析使用(e.g. 人口向量、改版矩陣...)
    2. QP使用的參數準備，儲存給matlab使用 (but resample的部分，人工計算(參考論文的論述))
)
'''

import os
import pandas as pd 
import numpy as np
from collections import Counter
from config import Config_simul

def read_simulation_data(dir_path_in):
    """
    Read simulation data from .csv files
    (讀取simulation資料
    讀取所有資料夾內的資料
    但是要依照序號排序好)
    Input: 
    :dir_path_in: Directory which contains generated simulation data, 
    with file name as 'year_0', 'year_1', 'year_2', ...
    Output: 
    :simulation_datta_df_list: DataFrame which simulation data store in 
    """
    file_list = os.listdir(dir_path_in) 
    file_list = [i for i in file_list if os.path.isfile(os.path.join(dir_path_in, i)) and os.path.join(dir_path_in, i).endswith('.csv')]
    # sort by number at the end of file name
    sorted_file_list = sorted(file_list, key=lambda a_file_nm: int(a_file_nm.replace('year_', '').replace('.csv', '')))

    file_path_list = [dir_path_in + i for i in sorted_file_list]

    simulation_datta_df_list = []
    for i in file_path_list: 
        simulation_datta_df_list.append(
            pd.read_csv(i).iloc[:, :]
        )
        
    return simulation_datta_df_list

def simulation_data_info_retrieve(datta_df_list):
    """
    Object: 
    Extract information from simulation data
    (
        (simulation資訊抽取) 
        所有可以從simulation data當中計算出來的資訊(e.g. 真實改版矩陣、真實習慣矩陣...)，在performence裡面的其中一個函數全部算完，並且全部存成csv檔案留作紀錄
        注意!! 不要把計算資訊放到data_gen當中做，data_gen只負責產生simulation資料而已

        (simulation資訊抽取)
        1.1 (全體) (cohort) 1997年的全體人口向量 (cohort的部分包含在QP參數準備當中)
        1.2 (全體) (cohort) 1998年的全體人口向量 
        1.3 (全體) (cohort) 1999年的全體人口向量 
        2. (全體) (cohort) 1997年的人口填寫1998年問卷向量 
        3. (全體) (cohort) 1998年的全體人口向量 (cohort的部分包含在QP參數準備當中)
        4. (全體) (cohort) 1997到1998年的改版矩陣 (cohort的部分包含在QP參數準備當中)
        5. (cohort) 1997到1998年的習慣矩陣
        6. (cohort) 1998到1999年的習慣矩陣 (cohort的部分包含在QP參數準備當中)
    )

    Input: 
    :datta_df_list: (list) raw simulation data(simulation的原始資料)

    Output: 
    :first_year_population_vec: (np.array) (4x1) population vector in 1997(1997年的全體人口向量)
    :second_year_population_vec: (np.array) (5x1) population vector in 1998(1998年的全體人口向量)
    :third_year_population_vec: (np.array) (5x1) population vector in 1999(1999年的全體人口向量)
    :first_yr_first_cohort_vec: (np.array) (4x1) cohort population vector in 1997(1997年的cohort人口向量)
    :second_yr_first_cohort_vec: (np.array) (5x1) cohort population vector in 1998(these cohort also exist in 1997)(1998年的cohort人口向量 (第一個cohort))
    :second_yr_second_cohort_vec: (np.array) (5x1) cohort population vector in 1998(these cohort also exist in 1999)(1998年的cohort人口向量 (第二個cohort))
    :third_yr_first_cohort_vec: (np.array) (5x1) cohort population vector in 1999(1999年的cohort人口向量)

    :first_year_population_second_year_q_vec: (np.array) (5x1) population vector in 1997 but fill 1998's questionnaire
    :first_cohort_second_year_q_vec: (np.array) (5x1) cohort vector in 1997 but fill 1998's questionnaire
    
    :first_cohort_habbit_matrix: (np.array) (5x5) underlying distribution change transition matrix of cohort between 1997 and 1998(1997到1998年的cohort人口習慣矩陣)
    :first_pop_ver_change_matrix: (np.array) (5x4) revision matrix of population sample between 1997 and 1998(1997到1998年的全體人口改版矩陣)
    :first_cohort_ver_change_matrix: (np.array) (5x4) revision matrix of cohort between 1997 and 1998(1997到1998年的cohort人口改版矩陣)
    """
    # 第一組cohort data 
    first_cohort_data = pd.merge(datta_df_list[0], datta_df_list[1], how='inner', on=['id'])
    # 第二組cohort data 
    second_cohort_data = pd.merge(datta_df_list[1], datta_df_list[2], how='inner', on=['id'])

    # 1.1 (全體) (cohort) 1997年的全體人口向量 (cohort的部分包含在QP參數準備當中)
    first_year_population_vec = datta_df_list[0]['final_q_result'].value_counts().sort_index()
    # print('first year population vec: \n', first_year_population_vec)
    # 1.2 (全體) 1998年的全體人口向量
    second_year_population_vec = datta_df_list[1]['final_q_result'].value_counts().sort_index()
    # 1.3 (全體) 1999年的全體人口向量
    third_year_population_vec = datta_df_list[2]['final_q_result'].value_counts().sort_index()

    # 1.4 (cohort) 第一組cohort第一年向量
    first_yr_first_cohort_vec = first_cohort_data['final_q_result_x'].value_counts().sort_index()
    # 1.5 (cohort) 第一組cohort第二年向量
    second_yr_first_cohort_vec = first_cohort_data['final_q_result_y'].value_counts().sort_index()
    # 1.6 (cohort) 第二組cohort第一年向量
    second_yr_second_cohort_vec = second_cohort_data['final_q_result_x'].value_counts().sort_index()
    # 1.7 (cohort) 第二組cohort第二年向量
    third_yr_first_cohort_vec = second_cohort_data['final_q_result_y'].value_counts().sort_index()

    # 2. (全體) (cohort) 1997年的人口填寫1998年問卷向量 (比例向量)
    first_year_population_second_year_q_vec = datta_df_list[0]['next_yr_ver_q_result'].value_counts().sort_index()
    first_cohort_second_year_q_vec = first_cohort_data['next_yr_ver_q_result_x'].value_counts().sort_index()
    
    # print('first year population second year questionnaire vec: \n', first_year_population_second_year_q_vec)
    # print('first year cohort second year questionnaire vec: \n', first_cohort_second_year_q_vec)

    # 4. (全體) (cohort) 1997到1998年的改版矩陣 (cohort的部分包含在QP參數準備當中)
    # (全體) 
    first_pop_ver_change_matrix = pd.crosstab(
        datta_df_list[0]['next_yr_ver_q_result'], 
        datta_df_list[0]['final_q_result'], 
        margins=True, 
        normalize='columns'
    )
    first_pop_ver_change_matrix = first_pop_ver_change_matrix.iloc[:, :-1]
    # print('第一個全體人口的改版矩陣: \n', first_pop_ver_change_matrix) 

    first_cohort_ver_change_matrix = pd.crosstab(
        first_cohort_data['next_yr_ver_q_result_x'], 
        first_cohort_data['final_q_result_x'], 
        margins=True, 
        normalize='columns'
    )
    first_cohort_ver_change_matrix = first_cohort_ver_change_matrix.iloc[:, :-1]
    # print('第一個cohort人口的改版矩陣: \n', first_cohort_ver_change_matrix)

    # 5.1 (cohort) 1997到1998年的習慣矩陣 
    first_cohort_habbit_matrix = pd.crosstab(
        first_cohort_data['final_q_result_y'], 
        first_cohort_data['next_yr_ver_q_result_x'], 
        margins=True, 
        normalize='columns'
    )
    first_cohort_habbit_matrix = first_cohort_habbit_matrix.iloc[:, :-1]
    # print('first cohort habbit matrix: \n', first_cohort_habbit_matrix)

    # 6. (cohort) 1998到1999年的習慣矩陣 (cohort的部分包含在QP參數準備當中) (全體人口目前還無法計算轉移矩陣) 

    return first_year_population_vec, second_year_population_vec, third_year_population_vec, \
        first_yr_first_cohort_vec, second_yr_first_cohort_vec, second_yr_second_cohort_vec, third_yr_first_cohort_vec, \
        first_year_population_second_year_q_vec, first_cohort_second_year_q_vec, \
        first_cohort_habbit_matrix, \
        first_pop_ver_change_matrix, first_cohort_ver_change_matrix

def QP_matrix_param_calculate(datta_df_list, main_directory):
    """
    Object: 
        Prepare vectors and matrices for QP to estimate revision matrix. 
        All the prepared data must output to /qp_input_output/ directory. 
        (QP的目標是估計第一和第二個年份間的改版矩陣)

    (
        1. (用手算) 針對第一年(97年)做gamma fit，並且重新取樣，然後作答第二年的問卷，在原本的問卷填答結果，產生改版矩陣的初始值
        2. 求出97年的人口向量 (重複健檢人口)
        3. 求出98年的人口向量 (重複健檢人口)
        4. 求出97年到98年的總變異矩陣
        5. 求出98年到99年的習慣矩陣
        6. 使用matlab QP求出，最佳的改版矩陣

        以上東西要全部輸出成.csv檔案，到./simulation_with_random_switch/qp_input_output 裡面
    )

    Input: 
    :datta_df_list: (list) (pd.DataFrame) 三年的資料表 
    :main_directory: (str) Address of the main directory, which includes a /qp_input_output/ folder 

    Ouput: 
    :python_G_matrix.csv: A hand craft ideal revision matrix between 1997 and 1998(97到98年預先估計的改版矩陣)
    :python_c_vec.csv: Relative frequency of cohort in 1997(97年的人數比例向量)
    :python_f_vec.csv: Relative frequency of cohort in 1998(98年的人數比例向量)
    :python_T_matrix.csv: Transition matrix of cohort between 1997 and 1998(97到98年的總變異矩陣)
    :python_M_matrix.csv: underlying distribution change matrix between 1998 and 1999(98到99年的習慣矩陣)
    """

    cohort_datta_df = pd.merge(datta_df_list[0], datta_df_list[1], how='inner', on=['id'])

    # 1. 求97年人口向量 (cohort)(重複健檢人口) (c vector)
    first_yr_cohort_vec = dict(Counter(cohort_datta_df['final_q_result_x']))
    first_yr_cohort_vec = {k: first_yr_cohort_vec[k] for k in sorted(first_yr_cohort_vec)}
    first_yr_cohort_vec_arr = np.array(list(first_yr_cohort_vec.values())) 
    first_yr_cohort_vec_number_sr = pd.Series(first_yr_cohort_vec_arr)  # 多存一個人數向量出來，matlab的是比例向量
    first_yr_cohort_vec_arr = first_yr_cohort_vec_arr / first_yr_cohort_vec_arr.sum()
    print('97年的人口向量: \n', first_yr_cohort_vec)
    print(np.array(list(first_yr_cohort_vec.values())))
    print(first_yr_cohort_vec_arr)
    first_yr_cohort_vec_df = pd.DataFrame({'vec': first_yr_cohort_vec_arr})
    first_yr_cohort_vec_df.to_csv(main_directory+'qp_input_output/python_c_vec.csv', index=False)

    # 2. 求98年的人口向量 (重複健檢人口) (f vector)
    second_yr_cohort_vec = dict(Counter(cohort_datta_df['final_q_result_y']))
    second_yr_cohort_vec = {k: second_yr_cohort_vec[k] for k in sorted(second_yr_cohort_vec)}
    second_yr_cohort_vec_arr = np.array(list(second_yr_cohort_vec.values()))
    second_yr_cohort_vec_number_sr = pd.Series(second_yr_cohort_vec_arr)
    second_yr_cohort_vec_arr = second_yr_cohort_vec_arr / second_yr_cohort_vec_arr.sum()
    print('98年的人口向量: \n', second_yr_cohort_vec)
    print(second_yr_cohort_vec_arr)
    second_yr_cohort_vec_df = pd.DataFrame({'vec': second_yr_cohort_vec_arr})
    second_yr_cohort_vec_df.to_csv(main_directory+'qp_input_output/python_f_vec.csv', index=False)

    # 3. 97到98年的總變異矩陣 (T matrix)
    cohort_datta_df = pd.merge(datta_df_list[0], datta_df_list[1], how='inner', on=['id'])
    first_total_transition_matrix = pd.crosstab(cohort_datta_df['final_q_result_y'], cohort_datta_df['final_q_result_x'], margins=True, normalize='columns')
    print('97到98年的總變異矩陣: \n', first_total_transition_matrix)
    first_total_transition_matrix.to_csv(main_directory+'qp_input_output/python_T_matrix.csv', index=False)

    # 4. 求出98到99年的習慣矩陣 (M matrix)
    second_cohort_datta_df = pd.merge(datta_df_list[1], datta_df_list[2], how='inner', on=['id'])
    second_habbit_transition_matrix = pd.crosstab(second_cohort_datta_df['final_q_result_y'], second_cohort_datta_df['final_q_result_x'], margins=True, normalize='columns')
    second_habbit_transition_matrix.to_csv(main_directory+'qp_input_output/python_M_matrix.csv', index=False)
    second_habbit_transition_matrix = second_habbit_transition_matrix.iloc[:, :-1]
    print('98到99年的習慣矩陣: \n', second_habbit_transition_matrix)

    return first_yr_cohort_vec_number_sr, second_yr_cohort_vec_number_sr, first_total_transition_matrix, second_habbit_transition_matrix

if __name__ == "__main__": 
    simulationConfig = Config_simul()
    simulation_data_threshold_list = simulationConfig.threshold_list
    simulation_datta_df_list = read_simulation_data(simulationConfig.main_directory) 

    # Create directory(建立資料夾)
    paths = [
        simulationConfig.main_directory + 'matrix_and_vector', 
        simulationConfig.main_directory + 'qp_input_output'
    ]
    for path in paths: 
        folder = os.path.exists(path)
        #判斷結果
        if not folder:
            #如果不存在，則建立新目錄
            os.makedirs(path)
            print('-----dir建立成功-----')

    # ======= ======= ======= ======= ======= ======= =======
    # 1. simulation資訊抽取
    first_year_population_vec, second_year_population_vec, third_year_population_vec, \
        first_yr_first_cohort_vec, second_yr_first_cohort_vec, second_yr_second_cohort_vec, third_yr_first_cohort_vec, \
        first_year_population_second_year_q_vec, first_cohort_second_year_q_vec, \
        first_cohort_habbit_matrix, \
        first_pop_ver_change_matrix, first_cohort_ver_change_matrix = \
            simulation_data_info_retrieve(simulation_datta_df_list)
    # 1.1 儲存成csv檔案
    first_year_population_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_year_population_vec.csv', index=False)
    first_year_population_second_year_q_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_year_population_second_year_q_vec.csv', index=False)
    first_cohort_second_year_q_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_cohort_second_year_q_vec.csv', index=False)
    second_year_population_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/second_year_population_vec.csv', index=False)
    third_year_population_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/third_year_population_vec.csv', index=False)
    first_cohort_habbit_matrix.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_cohort_habbit_matrix.csv', index=False)
    first_pop_ver_change_matrix.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_pop_ver_change_matrix.csv', index=False)
    first_cohort_ver_change_matrix.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_cohort_ver_change_matrix.csv', index=False)
    first_yr_first_cohort_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/first_yr_first_cohort_vec.csv', index=False)
    second_yr_first_cohort_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/second_yr_first_cohort_vec.csv', index=False)
    second_yr_second_cohort_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/second_yr_second_cohort_vec.csv', index=False)
    third_yr_first_cohort_vec.to_csv(simulationConfig.main_directory + 'matrix_and_vector/third_yr_first_cohort_vec.csv', index=False)



    # ======= ======= ======= ======= ======= ======= =======
    # 2. QP使用參數準備
    first_yr_cohort_vec, second_yr_cohort_vec, first_total_transition_matrix, second_habbit_transition_matrix = \
        QP_matrix_param_calculate(simulation_datta_df_list, simulationConfig.main_directory)
    print('第一年的cohort向量: \n', first_yr_cohort_vec)

    # 2.1 存成csv檔案
    first_yr_cohort_vec.to_csv(simulationConfig.main_directory + '/matrix_and_vector/first_yr_cohort_vec.csv', index=False)
    second_yr_cohort_vec.to_csv(simulationConfig.main_directory + '/matrix_and_vector/second_yr_cohort_vec.csv', index=False)
    first_total_transition_matrix.to_csv(simulationConfig.main_directory + '/matrix_and_vector/first_total_transition_matrix.csv', index=False)
    second_habbit_transition_matrix.to_csv(simulationConfig.main_directory + '/matrix_and_vector/second_habbit_transition_matrix.csv', index=False)
