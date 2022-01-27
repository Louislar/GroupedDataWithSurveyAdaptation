'''

要放在server上跑的code
Object: 使用gamma(或其他distribution)做resampling，並且計算"理論上"的改版矩陣，未來會當作QP的參數
        所以預計放在server上面做的事，有
        1. resampling (給定gamma的參數)
        2. 填第二年問卷
        3. 算出改版矩陣

'''

import os 
import numpy as np 
import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt 
from common_module_pkg import cdf_likelihood_study
from common_module_pkg import gamma_generater
from truncated_gamma_resample import resample_by_truncated_gamma_scipy

main_directory = './multiple_distribution_simulation_data/gamma/give_bias_distance_info/'

def read_simulation_data(dir_path_in):
    """
    讀取simulation資料
    讀取所有資料夾內的資料
    但是要依照序號排序好
    """
    file_list = os.listdir(dir_path_in) 
    file_list = [i for i in file_list if os.path.isfile(os.path.join(dir_path_in, i)) and os.path.join(dir_path_in, i).endswith('.csv')]
    # sort by number at the end of file name
    sorted_file_list = sorted(file_list, key=lambda a_file_nm: int(a_file_nm.replace('year_', '').replace('.csv', '')))

    file_path_list = [dir_path_in + i for i in sorted_file_list]
    # print(file_path_list)

    simulation_datta_df_list = []
    for i in file_path_list: 
        simulation_datta_df_list.append(
            pd.read_csv(i).iloc[:, :]
        )
        
    return simulation_datta_df_list

def QP_matrix_param_calculate(datta_df_list, threshold_list, second_yr_questionnaire):
    """
    Object: 
        估計第一和第二個年份間的改版矩陣

    針對第一年(97年)做gamma fit，並且重新取樣，然後作答第二年的問卷，在原本的問卷填答結果，產生改版矩陣的初始值

    以上東西要輸出成.csv檔案，到./simulation_with_random_switch/qp_input_output 裡面

    Input: 
    :datta_df_list: (list) (pd.DataFrame) 三年的資料表 
    :: 

    Ouput: 
    :python_G_matrix.csv: 97到98年預先估計的改版矩陣
    """

    cohort_datta_df = pd.merge(datta_df_list[0], datta_df_list[1], how='inner', on=['id'])

    # 1. 求出改版矩陣的初始值 (使用gamma作為假設) (G matrix) (至少要跑30分鐘)
    mle_result_set = cdf_likelihood_study.extimate_distribution_by_CDF_and_MLE(
        cohort_datta_df['final_q_result_x'], 
        'gamma', 
        threshold_list[0] 
    )
    print(mle_result_set[0])
    print(Counter(cohort_datta_df['final_q_result_x']))
    print(Counter(datta_df_list[0]['final_q_result']))
    # 重新取樣 (分段重新取樣!!)
    resample_result_df = resample_by_truncated_gamma_scipy(
        # datta_df_list[0]['final_q_result'], 
        cohort_datta_df['final_q_result_x'], 
        mle_result_set[0], 
        [0] + threshold_list[0] , 
        True
    )
    print(resample_result_df)
    # 重新填問卷
    resample_result_df['q_result'] = gamma_generater.fill_questionnaire(
            second_yr_questionnaire, 
            resample_result_df[1997], 
            print_log=False, 
            return_list=True
        )
    print(resample_result_df)
    # 使用新舊問卷結果，計算改版矩陣
    # 對舊的問卷填答結果排序
    cohort_datta_df = cohort_datta_df.sort_values(by=['final_q_result_x'])
    print(cohort_datta_df[['sample_x', 'final_q_result_x']])
    # 計算transition matrix
    # print(pd.crosstab(resample_result_df['q_result'], datta_df_list[0]['final_q_result'], margins=True))
    initial_ver_change_transition_matrix = pd.crosstab(resample_result_df['q_result'].values, cohort_datta_df['final_q_result_x'].values, margins=True, normalize='columns')
    print('transition matrix by gamma fit以及resampling: ')
    print(initial_ver_change_transition_matrix)
    initial_ver_change_transition_matrix.to_csv('./simulation_with_random_switch/qp_input_output/python_G_matrix.csv', index=False)


if __name__ == "__main__": 
    resample_distribution_nm = 'gamma'
    # resample_distribution_nm = 'exp'
    # resample_distribution_nm = 'weibull'
    # resample_distribution_nm = 'lognormal'
    simulation_data_threshold_list = [
        [1, 2, 3], 
        [1, 2.5, 4.5, 6.5], 
        [1, 2.5, 4.5, 6.5]
    ]

    # 讀進simulation資料
    simulation_datta_df_list = read_simulation_data(main_directory) 

    # 做resampling和計算改版矩陣
    # first_yr_cohort_vec_arr, second_yr_cohort_vec_arr, total_transition_matrix, habbit_transition_matrix = \
    # QP_matrix_param_calculate(simulation_datta_df_list, simulation_data_threshold_list, gamma_generater.questionnaire_1998)

    # 1. 算出cohort的人的資料
    cohort_datta_df = pd.merge(simulation_datta_df_list[0], simulation_datta_df_list[1], how='inner', on=['id'])
    
    # 2. 估計第一年人口的gamma參數
    mle_result_set = cdf_likelihood_study.extimate_distribution_by_CDF_and_MLE(
        cohort_datta_df['final_q_result_x'], 
        resample_distribution_nm, 
        simulation_data_threshold_list[0] 
    )
    distribution_fit_result_sr = pd.Series(mle_result_set[0])
    print(distribution_fit_result_sr)
    
    # 3. 儲存三個資訊成.csv
    # 3.1 第一個cohort的第一年問卷填答結果
    # 3.2 估計出來的gamma參數
    # 3.3 第一年的thresholds 
    cohort_datta_df['final_q_result_x'].to_csv(main_directory+'server_resample_prepare/first_cohort_first_q_vec.csv', index=False)
    distribution_fit_result_sr.to_csv(main_directory+'server_resample_prepare/distribution_fit_result_sr.csv', index=False)
    pd.Series(simulation_data_threshold_list[0]).to_csv(main_directory+'server_resample_prepare/second_yr_q_sr.csv', index=False)


