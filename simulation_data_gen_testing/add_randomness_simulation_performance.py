'''
使用simulaition出來的資料做預測，
使用兩種方法預測: 
1. 單純gamma fitting
2. QP matrix learning + [ gamma fitting or midpoint ]
'''

import os 
import numpy as np 
import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt 
import scipy.special as sp 
from common_module_pkg import cdf_likelihood_study

main_directory = './simul_data/'

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

def use_simple_gamma_fit(datta_df_list, thresholds_list, distribution_nm_list):
    """
    對每一個simulation出來的data set做估計/近似
    回傳近似出的結果: 
    1. gamma 參數
    2. (未完成) maximum likelihood 
    3. (未完成) AIC
    4. (未完成) BIC
    5. (未完成) chi square statistics
    """
    fitting_result_list = []
    for idx in range(len(datta_df_list)): 

        a_choice_data_sr = datta_df_list[idx]['final_q_result']

        mle_result_set = cdf_likelihood_study.extimate_distribution_by_CDF_and_MLE(
            a_choice_data_sr, 
            distribution_nm_list[idx], 
            thresholds_list[idx]
        ) 
        if distribution_nm_list[idx]=='gamma': 
            fitting_result_list.append([mle_result_set[0][0], mle_result_set[0][1]])
        elif distribution_nm_list[idx]=='exp': 
            fitting_result_list.append([mle_result_set[0][0]])
        elif distribution_nm_list[idx]=='weibull': 
            fitting_result_list.append([mle_result_set[0][0], mle_result_set[0][1]])
        elif distribution_nm_list[idx]=='lognormal': 
            fitting_result_list.append([mle_result_set[0][0], mle_result_set[0][1]])
        else: 
            raise EOFError

    return fitting_result_list 

def cal_simple_mid_point(datta_df_list, midpoint_list):
    """
    使用單純的取中點，計算年平均
    """
    fitting_result_list = []
    for idx in range(len(datta_df_list)): 

        a_choice_data_sr = datta_df_list[idx]['final_q_result']
        a_choice_data_count_dict = dict(sorted(dict(Counter(a_choice_data_sr) ).items(), key=lambda x: x[0]) )
        a_choice_data_count_arr = np.array([*a_choice_data_count_dict.values()])
        # print(a_choice_data_count_dict)
        # print(a_choice_data_count_arr)

        a_yr_mean = np.dot(a_choice_data_count_arr, midpoint_list[idx])
        a_yr_mean = a_yr_mean / sum(a_choice_data_count_arr)
        print(a_yr_mean)
        fitting_result_list.append(a_yr_mean)

    return fitting_result_list 

def load_matlab_QP_estimate_result(dir_path, file_nm=['matlab_version_change_matrix.csv', 'matlab_habbit_matrix.csv', 'matlab_new_cohort_97_vec.csv']): 
    """
    讀取matlab QP預測的結果

    Input: 
    :dir_path: (str) matlab輸出結果的資料夾位置
    :file_nm: (list) matlab輸出結果的檔案名稱 
                        依序的檔案名稱為: 
                        1. 估計的cohort 1997到1998的改版矩陣
                        2. 估計的cohort 1997到1998的習慣矩陣
                        3. 估計的cohort 1997年人口填寫1998年問卷的人口比例向量
    """
    file_path_list = [dir_path+i for i in file_nm]

    qp_result_list = [pd.read_csv(a_file_path, header=None) for a_file_path in file_path_list]

    # for i in qp_result_list: 
    #     print(i)

    return qp_result_list

def distribution_fit_by_sample_arr(sample_vec_arr, threshold_list, distribution_nm='gamma'):
    """
    使用內含人數的人口向量，近似給定的distribution
    ref: multi_year_patient_analysis.py 
    """
    # 建立資料 (之前在其他地方計算好了)
    # pre_cal_population_list = [10479, 4689, 2980, 1331, 2072] 
    pre_cal_population_list = sample_vec_arr
    population_choice_list = [[i+1]*pre_cal_population_list[i] for i in range(len(pre_cal_population_list))]
    population_choice_list2 = []
    for i in population_choice_list: 
        population_choice_list2 += i
    population_choice_arr = np.array(population_choice_list2)
    print(population_choice_arr[0])

    thresholds_1997 = [1.0, 2.0, 3.0] 
    thresholds_1998 = [1, 2.5, 4.5, 6.5] 

    param_list, max_log_likelihood, cumu_prob_list = \
        cdf_likelihood_study.extimate_distribution_by_CDF_and_MLE(
            population_choice_arr, 
            distribution_nm, 
            threshold_list 
        )
    
    print('best fit gamma param: ', param_list)
    return param_list

# corhort資料全部使用gamma預測一遍

# 將估計的人數取到整數
def make_population_vec_integer(population_vec: np.array, number_of_population: int):
    """
    將估計出來的人數轉換成整數，但是總數要符合給定的數值，並且要依照小數的大小作為優先加減的依據

    Input: 
    :population_vec: 估計出來帶有小數的人數比例向量
    :number_of_population: 總體人數數量
    """
    
    round_arr = np.floor(population_vec).astype(int)
    # print(round_arr)

    sum_of_round_arr = np.sum(round_arr).astype(int)
    # print(sum_of_round_arr)

    number_need_to_add = number_of_population - sum_of_round_arr
    print(number_of_population)
    print(number_need_to_add)

    decimal_arr = population_vec - round_arr
    # print(decimal_arr)

    sorted_decimal_array_idx = np.argsort(decimal_arr)[::-1]
    # print(sorted_decimal_array_idx)

    add_1_idx = sorted_decimal_array_idx[:number_need_to_add]
    # print(add_1_idx)
 
    round_arr[add_1_idx] += 1
    # print(round_arr)

    return round_arr 

if __name__ == "__main__":
    # 建立資料夾
    paths = [
        main_directory + 'data_for_draw_fig'
    ]
    for path in paths: 
        folder = os.path.exists(path)
        #判斷結果
        if not folder:
            #如果不存在，則建立新目錄
            os.makedirs(path)
            print('-----dir建立成功-----')
    # =======
    simulation_data_distribution_nm_list = ['gamma', 'gamma', 'gamma']
    # simulation_data_distribution_nm_list = ['exp', 'exp', 'exp']
    # simulation_data_distribution_nm_list = ['weibull', 'weibull', 'weibull']
    # simulation_data_distribution_nm_list = ['lognormal', 'lognormal', 'lognormal']
    simulation_data_threshold_list = [
        [1, 2, 3], 
        [1, 2.5, 4.5, 6.5], 
        [1, 2.5, 4.5, 6.5]
    ]
    questionnaire_value_1997 = [0.5, 1.5, 2.5, 3.0]
    questionnaire_value_1998 = [0.5, 1.75, 3.5, 5.5, 6.5]
    simulation_data_midpoint_list = [
        questionnaire_value_1997, 
        questionnaire_value_1998, 
        questionnaire_value_1998
    ]

    simulation_datta_df_list = read_simulation_data(main_directory) 

    # 1. 使用單純的gamma fit的結果
    simple_gamma_result_list = use_simple_gamma_fit(simulation_datta_df_list, simulation_data_threshold_list, simulation_data_distribution_nm_list)
    if simulation_data_distribution_nm_list[0]=='gamma': 
        simple_gamma_result_list = [param_list[0] * param_list[1] for param_list in simple_gamma_result_list]
    elif simulation_data_distribution_nm_list[0]=='exp': 
        simple_gamma_result_list = [1 / param_list[0] for param_list in simple_gamma_result_list]
    # r預測出來的weibull參數為 [0]: shape param 'a', [1]: scale param 'b', similar to python param. notation
    # So expect value will be: b * gamma_function(1 + 1 / a)
    elif simulation_data_distribution_nm_list[0]=='weibull': 
        simple_gamma_result_list = [param_list[1]*sp.gamma(1 + (1 / param_list[0])) for param_list in simple_gamma_result_list]
    # r預測出來的lognormal參數為 [0]: mu = (python) ln(scale), [1]: sd = (python) s
    # Expect value is exp(mu + (s^2)/2)
    elif simulation_data_distribution_nm_list[0]=='lognormal': 
        simple_gamma_result_list = [np.exp(param_list[0] + (param_list[1]**2)/2) for param_list in simple_gamma_result_list]

    print('(全體)單純gamma fit結果', simple_gamma_result_list)
    pd.Series(simple_gamma_result_list).to_csv(main_directory+'data_for_draw_fig/population_gamma_fit_mean.csv', index=False)
    

    # 2. 使用midpoint的結果 
    simple_gamma_result_list = cal_simple_mid_point(simulation_datta_df_list, simulation_data_midpoint_list)
    print('(全體)單純midpoint結果', simple_gamma_result_list)
    pd.Series(simple_gamma_result_list).to_csv(main_directory+'data_for_draw_fig/population_midpoint_mean.csv', index=False)

    # 8. 讀取matlab QP預測的結果 
    qp_result_list = load_matlab_QP_estimate_result(main_directory+'qp_input_output/')
    estimate_first_year_cohort_ver_change_matrix = qp_result_list[0]
    estimate_first_year_cohort_habbit_matrix = qp_result_list[1].values
    estimate_first_year_cohort_second_yr_q_vec_percent = qp_result_list[2].values[:, 0]
    print('估計的第一年cohort人口的習慣矩陣: \n', estimate_first_year_cohort_habbit_matrix)
    print('估計的第一年cohort人口填寫第二年問卷向量: \n', estimate_first_year_cohort_second_yr_q_vec_percent)

    # 8.1 使用matlab QP預測結果做計算
    # 8.1.1 計算估計的全體第一年人口填寫第二年問卷結果
    first_year_population_vec = pd.read_csv(main_directory+'matrix_and_vector/first_year_population_vec.csv')
    estimate_first_yr_population_second_yr_q_vec = \
        np.dot(estimate_first_year_cohort_ver_change_matrix.values, first_year_population_vec.values)[:, 0]
    print(estimate_first_year_cohort_ver_change_matrix)
    print(first_year_population_vec)
    print(first_year_population_vec.sum())
    print('估計的第一年總體人口填寫第二年問卷: \n', estimate_first_yr_population_second_yr_q_vec)
    print(sum(estimate_first_yr_population_second_yr_q_vec))

    # 8.1.2 將人數向量去除小數的部分
    # 預測的第一年總體人口填寫第二年問卷, 人數向量
    estimate_first_yr_population_second_yr_q_vec = \
        make_population_vec_integer(estimate_first_yr_population_second_yr_q_vec, first_year_population_vec.values.astype(int).sum())
    print(estimate_first_yr_population_second_yr_q_vec)
    print(sum(estimate_first_yr_population_second_yr_q_vec))
    pd.Series(estimate_first_yr_population_second_yr_q_vec).to_csv(main_directory+'matrix_and_vector/estimate_first_yr_population_second_yr_q_vec.csv', index=False)

    # 4.1 QP matrix + gamma fit 的結果
    best_fit_param = distribution_fit_by_sample_arr(estimate_first_yr_population_second_yr_q_vec, simulation_data_threshold_list[1], simulation_data_distribution_nm_list[0])
    population_matrix_and_gamma_fit_result_first_yr_mean = None
    if simulation_data_distribution_nm_list[0] == 'gamma': 
        population_matrix_and_gamma_fit_result_first_yr_mean = best_fit_param[0] * best_fit_param[1]
    elif simulation_data_distribution_nm_list[0] == 'exp': 
        population_matrix_and_gamma_fit_result_first_yr_mean = 1 / best_fit_param[0]
    elif simulation_data_distribution_nm_list[0] == 'weibull': 
        population_matrix_and_gamma_fit_result_first_yr_mean = best_fit_param[1]*sp.gamma(1 + (1 / best_fit_param[0]))
    elif simulation_data_distribution_nm_list[0] == 'lognormal': 
        population_matrix_and_gamma_fit_result_first_yr_mean = np.exp(best_fit_param[0] + (best_fit_param[1]**2)/2)
    print('全體人口1997年平均 (QP + {0} fit): '.format(simulation_data_distribution_nm_list[0]), population_matrix_and_gamma_fit_result_first_yr_mean)
    
    # 4.2 使用QP matrix + midpoint的結果
    # 先將人口向量換成人數比例向量
    estimate_first_yr_population_second_yr_q_vec = estimate_first_yr_population_second_yr_q_vec / np.sum(estimate_first_yr_population_second_yr_q_vec)
    population_matrix_and_midpoint_result_first_yr_mean = \
        np.dot(estimate_first_yr_population_second_yr_q_vec, questionnaire_value_1998)
    print('全體人口1997年平均 (QP + midpoint): ', population_matrix_and_midpoint_result_first_yr_mean)

    # 4.3 儲存全體人口QP預測結果
    estimate_first_year_population_mean_sr = \
        pd.Series([population_matrix_and_midpoint_result_first_yr_mean, population_matrix_and_gamma_fit_result_first_yr_mean])
    estimate_first_year_population_mean_sr.to_csv(main_directory+'data_for_draw_fig/population_qp_mean.csv', index=False)

    # ======= ======= ======= ======= ======= ======= =======
    # 6. 輸出corhort資料的預測結果 (預測的年平均)
    # 6.1 corhort的midpoint計算
    first_corhort_df = pd.merge(simulation_datta_df_list[0], simulation_datta_df_list[1], how='inner', on=['id'])
    first_yr_corhort_df = first_corhort_df[['id', 'final_q_result_x']].rename(columns={'final_q_result_x': 'final_q_result'})
    second_yr_corhort_df = first_corhort_df[['id', 'final_q_result_y']].rename(columns={'final_q_result_y': 'final_q_result'})
    # print(first_corhort_df)
    # print(first_yr_corhort_df)
    # print(second_yr_corhort_df)
    second_corhort_df = pd.merge(simulation_datta_df_list[1], simulation_datta_df_list[2], how='inner', on=['id'])
    third_yr_corhort_df = second_corhort_df[['id', 'final_q_result_x']].rename(columns={'final_q_result_x': 'final_q_result'})
    fourth_yr_corhort_df = second_corhort_df[['id', 'final_q_result_y']].rename(columns={'final_q_result_y': 'final_q_result'})

    corhort_data_df_list = [
        first_yr_corhort_df, 
        second_yr_corhort_df, 
        third_yr_corhort_df, 
        fourth_yr_corhort_df
    ]
    corhort_data_midpoint_list = [
        questionnaire_value_1997, 
        questionnaire_value_1998, 
        questionnaire_value_1998, 
        questionnaire_value_1998
    ]
    corhort_midpoint_result_list = cal_simple_mid_point(corhort_data_df_list, corhort_data_midpoint_list)
    # print(corhort_midpoint_result_list) # [1.558475, 2.277675, 2.1665, 2.2586375]
    cohort_midpoint_result_sr = pd.Series(corhort_midpoint_result_list)
    cohort_midpoint_result_sr.to_csv(main_directory+'data_for_draw_fig/cohort_midpoint_mean.csv', index=False)

    # 6.2 cohort的gamma fit計算
    cohort_data_threshold_list = [
        [1, 2, 3], 
        [1, 2.5, 4.5, 6.5], 
        [1, 2.5, 4.5, 6.5], 
        [1, 2.5, 4.5, 6.5]
    ]
    cohort_data_distribution_nm_list = [
        'gamma', 
        'gamma', 
        'gamma', 
        'gamma'
    ]
    # cohort_data_distribution_nm_list = [
    #     'exp', 
    #     'exp', 
    #     'exp', 
    #     'exp'
    # ]
    # cohort_data_distribution_nm_list = [
    #     'weibull', 
    #     'weibull', 
    #     'weibull', 
    #     'weibull'
    # ]
    # cohort_data_distribution_nm_list = [
    #     'lognormal', 
    #     'lognormal', 
    #     'lognormal', 
    #     'lognormal'
    # ]
    first_second_third_cohort_gamma_result_list = use_simple_gamma_fit(corhort_data_df_list, cohort_data_threshold_list, cohort_data_distribution_nm_list)
    if cohort_data_distribution_nm_list[0]=='gamma': 
        first_second_third_cohort_gamma_result_list = [a_set_param[0]*a_set_param[1] for a_set_param in first_second_third_cohort_gamma_result_list]
    elif cohort_data_distribution_nm_list[0]== 'exp': 
        first_second_third_cohort_gamma_result_list = [1 / a_set_param[0] for a_set_param in first_second_third_cohort_gamma_result_list]
    elif cohort_data_distribution_nm_list[0]== 'weibull': 
        first_second_third_cohort_gamma_result_list = [a_set_param[1]*sp.gamma(1 + (1 / a_set_param[0])) for a_set_param in first_second_third_cohort_gamma_result_list]
    elif cohort_data_distribution_nm_list[0]== 'lognormal': 
        first_second_third_cohort_gamma_result_list = [np.exp(a_set_param[0] + (a_set_param[1]**2)/2) for a_set_param in first_second_third_cohort_gamma_result_list]
    cohort_gamma_fit_result_sr = pd.Series(first_second_third_cohort_gamma_result_list)
    cohort_gamma_fit_result_sr.to_csv(main_directory+'data_for_draw_fig/cohort_gamma_fit_mean.csv', index=False)
    

    # 比例向量轉人數向量
    # 計算估計的第一年cohort人口填寫第二年問卷結果 (人數向量)
    print(estimate_first_year_cohort_second_yr_q_vec_percent)
    estimate_first_year_cohort_second_yr_q_vec = \
        estimate_first_year_cohort_second_yr_q_vec_percent * first_yr_corhort_df.shape[0]
    estimate_first_year_cohort_second_yr_q_vec = \
        make_population_vec_integer(estimate_first_year_cohort_second_yr_q_vec, first_yr_corhort_df.shape[0])
    print(estimate_first_year_cohort_second_yr_q_vec)

    # 6.3 cohort QP + gamma fit
    first_yr_corhort_QP_gamma_result = \
        distribution_fit_by_sample_arr(estimate_first_year_cohort_second_yr_q_vec, [1, 2.5, 4.5, 6.5], cohort_data_distribution_nm_list[0])
    if cohort_data_distribution_nm_list[0]=='gamma': 
        first_yr_corhort_QP_gamma_result = first_yr_corhort_QP_gamma_result[0] * first_yr_corhort_QP_gamma_result[1]
    elif cohort_data_distribution_nm_list[0]=='exp': 
        first_yr_corhort_QP_gamma_result = 1 / first_yr_corhort_QP_gamma_result[0] 
    elif cohort_data_distribution_nm_list[0]=='weibull': 
        first_yr_corhort_QP_gamma_result = first_yr_corhort_QP_gamma_result[1]*sp.gamma(1 + (1 / first_yr_corhort_QP_gamma_result[0]))
    elif cohort_data_distribution_nm_list[0]=='lognormal': 
        first_yr_corhort_QP_gamma_result = np.exp(first_yr_corhort_QP_gamma_result[0] + (first_yr_corhort_QP_gamma_result[1]**2)/2)


    # 6.4 cohort QP + midpoint 
    first_yr_corhort_QP_midpoint_result = \
        np.dot(estimate_first_year_cohort_second_yr_q_vec_percent, questionnaire_value_1998)

    cohort_qp_mean = pd.Series([first_yr_corhort_QP_midpoint_result, first_yr_corhort_QP_gamma_result])
    cohort_qp_mean.to_csv(main_directory+'data_for_draw_fig/cohort_qp_mean.csv', index=False)

