'''
讀取計算好的年平均檔案，拿來畫圖
1. 讀取計算好的年平均資料 (6個年平均資料檔)
1.1. 全體 三年的midpoint年平均
1.2. 全體 三年的gamma fit年平均
1.3. 全體 第一年的QP+gamma fit年平均 和 全體 第一年的QP+midpoint年平均
1.4. cohort 三年的midpoint年平均
1.5. cohort 三年的gamma fit年平均
1.6. cohort 第一年的QP+gamma fit年平均 和 cohort 第一年的QP+midpoint年平均

2. 讀取的資料繪圖
'''

import os 
import numpy as np 
import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt 
from add_randomness_simulation_performance import read_simulation_data

main_directory = './simul_data/'
origin_population_yearly_mean_list = [3, 2.8, 2.7] 




def read_yearly_mean(file_dir):
    """
    讀取計算好的年平均
    """
    population_gamma_fit_mean = pd.read_csv(file_dir+'population_gamma_fit_mean.csv')
    population_midpoint_mean = pd.read_csv(file_dir+'population_midpoint_mean.csv')
    population_qp_mean = pd.read_csv(file_dir+'population_qp_mean.csv')

    cohort_gamma_fit_mean = pd.read_csv(file_dir+'cohort_gamma_fit_mean.csv')
    cohort_midpoint_mean = pd.read_csv(file_dir+'cohort_midpoint_mean.csv')
    cohort_qp_mean = pd.read_csv(file_dir+'cohort_qp_mean.csv')

    return population_gamma_fit_mean, population_midpoint_mean, population_qp_mean, cohort_gamma_fit_mean, cohort_midpoint_mean, cohort_qp_mean

def cal_sample_mean(sample_df_list):
    """
    計算樣本年平均數
    """
    return [sample_df_list[0]['sample'].mean(), sample_df_list[1]['sample'].mean(), sample_df_list[2]['sample'].mean()]

def cal_cohort_sample_mean(sample_df_list): 
    '''
    計算cohrot樣本平均數
    '''
    first_cohort_df = pd.merge(simulation_datta_df_list[0], simulation_datta_df_list[1], how='inner', on=['id'])
    second_cohort_df = pd.merge(simulation_datta_df_list[1], simulation_datta_df_list[2], how='inner', on=['id'])
    return [first_cohort_df['sample_x'].mean(), first_cohort_df['sample_y'].mean(), second_cohort_df['sample_x'].mean(), second_cohort_df['sample_y'].mean()]

def draw_population_result(yr_mean_origin_list, yr_sample_mean_list, yr_mean_gamma_fit_list, yr_mean_midpoint_list, predict_first_yr_list):
    """
    畫出總體人口的結果
    Input: 
    :predict_first_yr_list: (list) 預測第一年的平均，依次使用的方法為，
                            QP+gamma, QP+midpoint
    """
    x = range(1, 3+1)
    plt.figure()
    plt.plot(x, yr_mean_origin_list, '-', label='origin')
    plt.plot(x, yr_sample_mean_list, '-', label='sample')
    plt.plot(x, yr_mean_gamma_fit_list, '-', label='gamma fit')
    plt.plot(x, yr_mean_midpoint_list, '-', label='mid point')
    plt.plot(1, predict_first_yr_list[0], '.r', markersize=20, label='matrix (mid point)')
    plt.plot(1, predict_first_yr_list[1], '.k', markersize=20, label='matrix (gamma)')

    # 標記估計出來的點的數值
    plt.annotate(text=str(round(predict_first_yr_list[0][0], 5)), xy=(1.1, predict_first_yr_list[0]))
    plt.annotate(text=str(round(predict_first_yr_list[1][0], 5)), xy=(1.1, predict_first_yr_list[1]))

    plt.xticks(x)
    plt.legend()
    # plt.show()

def draw_corhort_result(yr_sample_mean_list, yr_mean_gamma_fit_list, yr_mean_midpoint_list, predict_first_yr_list):
    """
    畫出重複健檢人口的結果
    """
    x = [1, 2, 2, 3]
    plt.figure()
    plt.plot(x, yr_sample_mean_list, '.--', markersize=20, label='sample')
    plt.plot(x, yr_mean_gamma_fit_list, '.--', markersize=20, label='gamma fit')
    plt.plot(x, yr_mean_midpoint_list, '.--', markersize=20, label='mid point')
    plt.plot(1, predict_first_yr_list[0], '.r', markersize=20, label='matrix (mid point)')
    plt.plot(1, predict_first_yr_list[1], '.k', markersize=20, label='matrix (gamma)')

    # 標記估計出來的點的數值
    plt.annotate(text=str(round(predict_first_yr_list[0][0], 5)), xy=(1.1, predict_first_yr_list[0]))
    plt.annotate(text=str(round(predict_first_yr_list[1][0], 5)), xy=(1.1, predict_first_yr_list[1]))

    plt.xticks(x)
    plt.legend()
    # plt.show()

def draw_population_percent(three_yr_pop_vec_arr: np.array, estimate_pop_vec: np.array):
    """
    畫出總體人口比例變化
    第一年使用填寫第二年問卷版本的向量
    預測的第一年另外點出來

    Input: 
    :three_yr_pop_vec_list: 三年人口數量的向量
    :estimate_pop_vec: 預測的第一年人口數量向量
    """
    three_yr_pop_vec_arr = three_yr_pop_vec_arr.astype(float)
    estimate_pop_vec = estimate_pop_vec.astype(float)
    # 人口數量轉成人口比例向量
    for i in range(len(three_yr_pop_vec_arr)): 
        three_yr_pop_vec_arr[i, :] = three_yr_pop_vec_arr[i, :] / three_yr_pop_vec_arr[i, :].sum()
    estimate_pop_vec = estimate_pop_vec / estimate_pop_vec.sum()
    print(three_yr_pop_vec_arr)
    print(list(zip(*three_yr_pop_vec_arr)))
    each_choice_percent_list = list(zip(*three_yr_pop_vec_arr))

    x = [1, 2, 3]
    x_estimate = [1, 1, 1, 1, 1]
    # plt.figure()
    choice_count = 1
    plot_list = []
    plt.figure()
    for a_choice_yearly_percent in each_choice_percent_list: 
        tmp_plot = plt.plot(x, a_choice_yearly_percent, '-', label='choice: {0}'.format(choice_count))
        plot_list.append(tmp_plot[0])
        choice_count += 1
    
    # 畫出估計的結果
    for i in range(len(estimate_pop_vec)): 
        plt.plot(1, estimate_pop_vec[i], '.', color=plot_list[i].get_color(), markersize=20, label='estimate') 
    
    plt.xticks(x)
    plt.legend()

if __name__ == "__main__": 

    # 讀取資料
    population_gamma_fit_mean, population_midpoint_mean, population_qp_mean, cohort_gamma_fit_mean, cohort_midpoint_mean, cohort_qp_mean = \
         read_yearly_mean(main_directory+'data_for_draw_fig/')
    first_year_population_second_year_q_vec = pd.read_csv(main_directory+'matrix_and_vector/first_year_population_second_year_q_vec.csv')
    second_year_population_vec = pd.read_csv(main_directory+'matrix_and_vector/second_year_population_vec.csv')
    third_year_population_vec = pd.read_csv(main_directory+'matrix_and_vector/third_year_population_vec.csv')
    estimate_first_yr_population_second_yr_q_vec = pd.read_csv(main_directory+'matrix_and_vector/estimate_first_yr_population_second_yr_q_vec.csv')

    # 計算樣本平均
    simulation_datta_df_list = read_simulation_data(main_directory)
    sample_mean_list = cal_sample_mean(simulation_datta_df_list)
    # 計算cohort樣本平均
    cohort_sample_mean_list = cal_cohort_sample_mean(simulation_datta_df_list)
    print(sample_mean_list)
    print(cohort_sample_mean_list)

    # 畫population趨勢圖
    draw_population_result(
        origin_population_yearly_mean_list, 
        sample_mean_list, 
        population_gamma_fit_mean.values, 
        population_midpoint_mean.values, 
        population_qp_mean.values
    )

    # 畫cohort趨勢圖
    draw_corhort_result(
        cohort_sample_mean_list,
        cohort_gamma_fit_mean.values, 
        cohort_midpoint_mean.values, 
        cohort_qp_mean.values
    )

    # 畫全體人口比例圖
    draw_population_percent(
        np.array([first_year_population_second_year_q_vec.values[:, 0], second_year_population_vec.values[:, 0], third_year_population_vec.values[:, 0]]), 
        estimate_first_yr_population_second_yr_q_vec.values[:, 0]
    )

    plt.show()
    