'''
比較各種向量與矩陣的差異

'''

import os 
import numpy as np 
import pandas as pd 
from collections import Counter 
import matplotlib.pyplot as plt 

# main_directory = './simulation_with_random_switch/thesis_with_or_without_mat_dis_info/'
main_directory = './multiple_distribution_simulation_data/gamma/no_distribution_assumption/'


# (QP估計結果與simulation資訊的比較)
# 目的: 我預測的準確嗎?
# 承接matlab輸出的QP估計結果，計算出我想要的結果
# 1. 估計的全體的1997年填寫1998年問卷人口比例向量 
# 2. 估計的cohort的1997年填寫1998年問卷人口比例向量 
# 3. 改版矩陣的距離 (估計的、真實的) --> 真實的改版矩陣需要用cohort資料計算
# 4. 習慣矩陣的距離 (估計的、真實的、給定的、98年的) --> 真實的習慣矩陣需要用cohort資料計算、還有給定的習慣矩陣可以加入比較
def compare_simulation_data_and_QP_result(
    matrix_distance_measurement_func, 
    vector_distance_measurement_func, 
    first_year_pop_second_year_q_vec, estimate_first_year_pop_second_year_q_vec, 
    first_year_cohort_second_year_q_vec, estimate_first_year_cohort_second_year_q_vec, 
    first_pop_ver_change_matrix, first_cohort_ver_change_matrix, estimate_first_ver_change_matrix, 
    first_cohort_habbit_matrix, estimate_first_cohort_habbit_matrix, 
    second_cohort_habbit_matrix, 
    first_year_pop_vec, first_cohort_vec
):
    """
    Object: 
    (QP估計結果與simulation資訊的比較) 再來，拿取計算好的資訊以及QP的估計結果，就可以算出我要的資訊 (e.g. 改版矩陣的距離、習慣矩陣的距離、人口向量的距離...)
    
    注意: 之前做的simulation資訊抽取，完全是為了QP計算的參數使用，所以要跟上面的TODO的function分開來實作，因為目的不同

    (QP估計結果與simulation資訊的比較)
    # 承接matlab輸出的QP估計結果，計算出我想要的結果
    # 1. 估計的全體的1997年填寫1998年問卷人口比例向量 
    # 2. 估計的cohort的1997年填寫1998年問卷人口比例向量 
    # 3. 全體人口與cohort人口的差距
    # 4. 改版矩陣的距離 (估計的、真實的) --> 真實的改版矩陣需要用cohort資料計算
    # 5. 習慣矩陣的距離 (估計的、真實的、給定的、98年的) --> 真實的習慣矩陣需要用cohort資料計算、還有給定的習慣矩陣可以加入比較

    Input: 
    :matrix_distance_measurement_func: (func) 衡量矩陣距離的函數
    :vector_distance_measurement_func: (func) 衡量向量距離的函數
    :first_year_pop_second_year_q_vec: (np.array) 第一年人口填寫第二年問卷的人口比例向量
    :estimate_first_year_pop_second_year_q_vec: (np.array) 估計的第一年人口填寫第二年問卷的人口比例向量
    :first_year_cohort_second_year_q_vec: (np.array) 第一年cohort人口填寫第二年問卷的人口比例向量
    :estimate_first_year_cohort_second_year_q_vec: (np.array) 估計第一年cohort人口填寫第二年問卷的人口比例向量
    :first_pop_ver_change_matrix: (np.array) (全體) 第一個改版矩陣 
    :first_cohort_ver_change_matrix: (np.array) (cohort) 第一個改版矩陣 
    :estimate_first_ver_change_matrix: (np.array) 估計的第一個改版矩陣
    :first_cohort_habbit_matrix: (np.array) (cohort) 第一個習慣矩陣
    :estimate_first_cohort_habbit_matrix: (np.array) (cohort) 估計的第一個習慣矩陣
    :second_cohort_habbit_matrix: (np.array) 第二個習慣矩陣
    :first_year_pop_vec: (np.array) 第一年的全體人口向量
    :first_cohort_vec: (np.array) 第一年的cohort人口向量 

    Output: 
    :diff_first_year_pop_second_year_q: (np.array) 估計與真實的差距: "第一年人口填寫第二年問卷的人口比例向量"
    :diff_first_cohort_second_q_vec: (np.array) 估計與真實的差距: "第一年cohort人口填寫第二年問卷的人口比例向量"
    :diff_first_year_pop_first_cohort: (np.array) 第一年全體人口向量 與 第一年cohort人口差異
    :diff_first_year_pop_first_cohort_second_year_q: (np.array) 第一年全體人口向量 與 第一年cohort人口差異 (都是填寫第二年問卷)
    :diff_first_ver_change_matrix: (np.array) 估計與真實的差距: "第一個改版矩陣"
    :diff_first_habbit_matrix: (np.array) 估計與真實的差距: "第一個習慣矩陣"
    :diff_second_habbit_matrix_estimated_first_habbit_matrix: (np.array) 第二個習慣矩陣 與 估計的第一個習慣矩陣
    :diff_first_habbit_matrix_second_habbit_matrix: (np.array) 第一個習慣矩陣 與 第二個習慣矩陣
    """
    
    # 1. 97總體人口填寫98問卷向量的預測準確程度
    diff_first_year_pop_second_year_q = vector_distance_measurement_func(first_year_pop_second_year_q_vec, estimate_first_year_pop_second_year_q_vec)

    # 2. 97 cohort人口填寫98問卷向量的預測準確程度
    diff_first_cohort_second_q_vec = \
        vector_distance_measurement_func(first_year_cohort_second_year_q_vec, estimate_first_year_cohort_second_year_q_vec)

    # 3. 97全體人口 與 97 cohort人口差異
    #      有兩種: 97問卷版本以及98問卷版本
    diff_first_year_pop_first_cohort = \
        vector_distance_measurement_func(first_year_pop_vec, first_cohort_vec)
    diff_first_year_pop_first_cohort_second_year_q = \
        vector_distance_measurement_func(first_year_pop_second_year_q_vec, first_year_cohort_second_year_q_vec)

    # 4. 改版矩陣的預測準確程度 (正確答案有兩種: cohort和全體) (估計只會有一種，因為我假設cohort與全體的改版矩陣非常相似)
    diff_first_pop_ver_change_matrix = matrix_distance_measurement_func(first_pop_ver_change_matrix, estimate_first_ver_change_matrix)
    diff_first_cohort_ver_change_matrix = matrix_distance_measurement_func(first_cohort_ver_change_matrix, estimate_first_ver_change_matrix)

    # 5. 習慣矩陣的預測準確程度
    # cohort習慣矩陣，真實97年與估計的差距
    diff_first_habbit_matrix = matrix_distance_measurement_func(first_cohort_habbit_matrix, estimate_first_cohort_habbit_matrix)
    # cohort習慣矩陣，真實98年與估計的差距
    diff_second_habbit_matrix_estimated_first_habbit_matrix = \
        matrix_distance_measurement_func(second_cohort_habbit_matrix, estimate_first_cohort_habbit_matrix)
    # cohort習慣矩陣，真實97年與真實98年的差距
    diff_first_habbit_matrix_second_habbit_matrix = \
        matrix_distance_measurement_func(first_cohort_habbit_matrix, second_cohort_habbit_matrix)


    return diff_first_year_pop_second_year_q, diff_first_cohort_second_q_vec, diff_first_year_pop_first_cohort, diff_first_year_pop_first_cohort_second_year_q, \
        diff_first_pop_ver_change_matrix, diff_first_cohort_ver_change_matrix, diff_first_habbit_matrix, diff_second_habbit_matrix_estimated_first_habbit_matrix, diff_first_habbit_matrix_second_habbit_matrix






# 9. QP結果與simulation資訊比較
    # diff_first_year_pop_second_year_q, diff_first_cohort_second_q_vec, diff_first_year_pop_first_cohort, diff_first_year_pop_first_cohort_second_year_q, \
    #     diff_first_pop_ver_change_matrix, diff_first_cohort_ver_change_matrix, diff_first_habbit_matrix, diff_second_habbit_matrix_estimated_first_habbit_matrix, diff_first_habbit_matrix_second_habbit_matrix = \
    #         compare_simulation_data_and_QP_result(
    #             max_absolute_difference_between_matrix, 
    #             max_absolute_difference_between_vector, 
    #             first_year_population_second_year_q_vec_percent, 
    #             estimate_first_yr_population_second_yr_q_vec, 
    #             first_cohort_second_year_q_vec_percent, 
    #             estimate_first_year_cohort_second_yr_q_vec_percent, 
    #             first_pop_ver_change_matrix, 
    #             first_cohort_ver_change_matrix,  
    #             estimate_first_year_cohort_ver_change_matrix, 
    #             first_cohort_habbit_matrix, 
    #             estimate_first_year_cohort_habbit_matrix, 
    #             habbit_transition_matrix, 
    #             first_year_population_vec, 
    #             first_yr_cohort_vec_arr
    #         )
    # print('估計與真實的差距: "第一年全體人口填寫第二年問卷的人口比例向量": ', diff_first_year_pop_second_year_q)
    # print('估計與真實的差距: "第一年cohort人口填寫第二年問卷的人口比例向量": ', diff_first_cohort_second_q_vec)


# ======= ======= ======= ======= ======= ======= =======
# 計算矩陣間距離的函數 start
 
def KL_divergence_between_vector(a_vector, b_vector):
    """
    計算向量間的KL divergence
    KL(a, b)
    """
    after_log = np.log(np.divide(a_vector, b_vector))
    after_log = np.where(np.isneginf(after_log), 0, after_log)  # 處理a column中有0的情況，要回傳0，而不是nan
    expect_value = np.sum(np.multiply(a_vector, after_log))
    # neg_expect_value = -1 * expect_value
    return expect_value

def KL_divergence_between_matrix(a_matrix, b_matrix):
    """
    計算轉移矩陣間的KL divergence，每一個column分開計算，然後取最大的
    這邊假設matrix都是column加總為1
    KL(a, b)
    """
    KL_list = [] 
    for i in range(a_matrix.shape[1]): 
        a_col = a_matrix[:, i]
        b_col = b_matrix[:, i]
        after_log_col = np.log(np.divide(a_col, b_col) )
        after_mul = np.multiply(a_col, after_log_col) 
        after_mul = np.where(np.isnan(after_mul), 0, after_mul)  # 處理a column中有0的情況，要回傳0，而不是nan
        KL_list.append(np.sum(after_mul)) 

    # print(KL_list)
    KL_list = np.array(KL_list)
    KL_idx_list = np.argsort(np.abs(KL_list))
    # print(sorted(KL_list)[0])
    return KL_list[KL_idx_list][-1] 

def max_absolute_difference_between_vector(a_vector, b_vector):
    """
    計算向量間的absolute difference，每一個cell/entry分開計算，然後取最大的
    兩個向量大小要相同
    """
    a_minus_b_abs = np.abs(np.subtract(a_vector, b_vector))
    max_absolute_difference = np.amax(a_minus_b_abs)
    return max_absolute_difference

def max_absolute_difference_between_matrix(a_matrix: np.array, b_matrix: np.array):
    """
    計算轉移矩陣間的absolute difference，每一個cell/entry分開計算，然後取最大的
    這邊假設matrix都是column加總為1
    兩個矩陣大小要相同
    """
    a_minus_b_abs = np.abs(np.subtract(a_matrix, b_matrix))
    max_absolute_difference = np.amax(a_minus_b_abs)
    # for i in a_matrix.shape[0]: 
    #     for j in a_matrix.shape[1]: 
    #         if max_absolute_difference < np.abs(a_matrix[i, j] - b_matrix[i, j]): 
    #             max_absolute_difference = np.abs(a_matrix[i, j] - b_matrix[i, j])
    
    return max_absolute_difference

def max_divide_difference_between_vector(a_vector, b_vector): 
    a_divide_b = np.divide(a_vector, b_vector)
    a_divide_b_adj = np.where(a_divide_b>=1, a_divide_b, 1.0/a_divide_b)
    max_diff = np.amax(a_divide_b_adj)
    return max_diff

def max_divide_difference_between_matrix(a_matrix: np.array, b_matrix: np.array): 
    a_divide_b = np.divide(a_matrix, b_matrix)
    a_divide_b_adj = np.where(a_divide_b>=1, a_divide_b, 1.0/a_divide_b)
    max_diff = np.amax(a_divide_b_adj)
    return max_diff


# 計算矩陣間距離的函數 end 
# ======= ======= ======= ======= ======= ======= =======


# 暫時使用
if __name__ == "__main__": 
    '''
    1.1 估計的97年cohort向量 與 正確的97年cohort向量
    1.2 估計的97年population向量 與 正確的97年population向量
    2. 估計的改版矩陣 與 正確的改版矩陣
    3.1 估計的習慣矩陣 與 正確的第一年習慣矩陣
    3.2 估計的習慣矩陣 與 正確的第二年習慣矩陣
    4. 正確的第一年的習慣矩陣 與 正確的第二年的習慣矩陣
    5. 估計的總變異矩陣 與 正確的總變異矩陣
    '''
    def read_all_data_needed(data_dir): 
        '''
        取所有需要的資料
        '''
        # 1.1.1 估計的97年cohort向量
        estimate_first_yr_cohort_vec = pd.read_csv(main_directory+'qp_input_output/matlab_new_cohort_97_vec.csv', header=None).values
        # 1.1.2 正確的97年cohort向量
        real_first_yr_cohort_vec = pd.read_csv(main_directory+'matrix_and_vector/first_cohort_second_year_q_vec.csv')
        real_first_yr_cohort_vec /= real_first_yr_cohort_vec.sum()
        real_first_yr_cohort_vec = real_first_yr_cohort_vec.values

        # 1.2.1 估計的97年population向量
        estimate_first_yr_population_vec = pd.read_csv(main_directory+'matrix_and_vector/estimate_first_yr_population_second_yr_q_vec.csv')
        estimate_first_yr_population_vec /= estimate_first_yr_population_vec.sum()
        estimate_first_yr_population_vec = estimate_first_yr_population_vec.values
        # 1.2.2 正確的97年population向量
        real_first_yr_population_vec = pd.read_csv(main_directory+'matrix_and_vector/first_year_population_second_year_q_vec.csv')
        real_first_yr_population_vec /= real_first_yr_population_vec.sum()
        real_first_yr_population_vec = real_first_yr_population_vec.values
        
        # 2.1 估計的改版矩陣
        estimate_ver_change_mat = pd.read_csv(main_directory+'qp_input_output/matlab_version_change_matrix.csv', header=None).values
        # 2.2 正確的改版矩陣
        real_ver_change_mat = pd.read_csv(main_directory+'matrix_and_vector/first_cohort_ver_change_matrix.csv').values

        # 3.1 估計的習慣矩陣
        estimate_first_habbit_mat = pd.read_csv(main_directory+'qp_input_output/matlab_habbit_matrix.csv', header=None).values
        # 3.2 正確的第一年習慣矩陣
        real_first_cohort_habbit_mat = pd.read_csv(main_directory+'matrix_and_vector/first_cohort_habbit_matrix.csv').values
        # 3.3 正確的第二年習慣矩陣
        real_second_cohort_habbit_mat = pd.read_csv(main_directory+'matrix_and_vector/second_habbit_transition_matrix.csv').values

        # 5.1 估計的總變異矩陣
        estimate_total_transition_mat = np.dot(estimate_first_habbit_mat, estimate_ver_change_mat)
        # 5.2 正確的總變異矩陣
        real_total_transition_mat = pd.read_csv(main_directory+'matrix_and_vector/first_total_transition_matrix.csv').values
        real_total_transition_mat = real_total_transition_mat[:, :-1]

        return [
            estimate_first_yr_cohort_vec, real_first_yr_cohort_vec, estimate_first_yr_population_vec, real_first_yr_population_vec, 
            estimate_ver_change_mat, real_ver_change_mat, 
            estimate_first_habbit_mat, real_first_cohort_habbit_mat, real_second_cohort_habbit_mat, estimate_total_transition_mat, 
            real_total_transition_mat
        ]
    
    def cal_distance(data_list): 
        '''
        1.1 估計的97年cohort向量 與 正確的97年cohort向量
        1.2 估計的97年population向量 與 正確的97年population向量
        2. 估計的改版矩陣 與 正確的改版矩陣
        3.1 估計的習慣矩陣 與 正確的第一年習慣矩陣
        3.2 估計的習慣矩陣 與 正確的第二年習慣矩陣
        3.3 正確的第一年的習慣矩陣 與 正確的第二年的習慣矩陣
        4. 估計的總變異矩陣 與 正確的總變異矩陣
        '''
        # 1.1 
        L_inf_dis_1_1 = max_absolute_difference_between_vector(data_list[1], data_list[0])
        KL_dis_1_1 = KL_divergence_between_vector(data_list[1], data_list[0])
        # 1.2 
        L_inf_dis_1_2 = max_absolute_difference_between_vector(data_list[3], data_list[2])
        KL_dis_1_2 = KL_divergence_between_vector(data_list[3], data_list[2])
        # 2. 
        L_inf_dis_2 = max_absolute_difference_between_matrix(data_list[5], data_list[4])
        KL_dis_2 = KL_divergence_between_matrix(data_list[5], data_list[4])
        # 3.1 
        L_inf_dis_3_1 = max_absolute_difference_between_matrix(data_list[7], data_list[6])
        KL_dis_3_1 = KL_divergence_between_matrix(data_list[7], data_list[6]) 
        # 3.2 
        L_inf_dis_3_2 = max_absolute_difference_between_matrix(data_list[8], data_list[6])
        KL_dis_3_2 = KL_divergence_between_matrix(data_list[8], data_list[6]) 
        # 3.3
        L_inf_dis_3_3 = max_absolute_difference_between_matrix(data_list[8], data_list[7])
        KL_dis_3_3_1 = KL_divergence_between_matrix(data_list[8], data_list[7]) 
        KL_dis_3_3_2 = KL_divergence_between_matrix(data_list[7], data_list[8]) 
        # 4
        L_inf_dis_4 = max_absolute_difference_between_matrix(data_list[10], data_list[9])
        KL_dis_4 = KL_divergence_between_matrix(data_list[10], data_list[9]) 

        print('1.1 估計的97年cohort向量 與 正確的97年cohort向量')
        print(L_inf_dis_1_1, ', ', KL_dis_1_1)
        print('\n1.2 估計的97年population向量 與 正確的97年population向量')
        print(L_inf_dis_1_2, ', ', KL_dis_1_2)
        print('\n2. 估計的改版矩陣 與 正確的改版矩陣')
        print(L_inf_dis_2, ', ', KL_dis_2)
        print('\n3.1 估計的習慣矩陣 與 正確的第一年習慣矩陣')
        print(L_inf_dis_3_1, ', ', KL_dis_3_1)
        print('\n3.2 估計的習慣矩陣 與 正確的第二年習慣矩陣')
        print(L_inf_dis_3_2, ', ', KL_dis_3_2)
        print('\n3.3 正確的第一年的習慣矩陣 與 正確的第二年的習慣矩陣')
        print(L_inf_dis_3_3, ', ', KL_dis_3_3_1, ', ', KL_dis_3_3_2)
        print('\n5. 估計的總變異矩陣 與 正確的總變異矩陣')
        print(L_inf_dis_4, ', ', KL_dis_4)

        # print('======= L infinity =======') 
        # print('======= KL divergence =======') 

    read_in_data_list = read_all_data_needed(main_directory)
    cal_distance(read_in_data_list)


if __name__ == "__main01__": 
    # 第一年習慣矩陣與第二年習慣矩陣的差異
    first_cohort_habbit_matrix = pd.read_csv(main_directory+'matrix_and_vector/first_cohort_habbit_matrix.csv').values
    second_cohort_habbit_matrix = pd.read_csv(main_directory+'qp_input_output/python_M_matrix.csv').values[:, :-1]
    print(first_cohort_habbit_matrix)
    print(second_cohort_habbit_matrix)
    print(max_absolute_difference_between_matrix(first_cohort_habbit_matrix, second_cohort_habbit_matrix))

    # 第一年習慣矩陣估計與真實的差異
    estimate_first_cohort_habbit_matrix = pd.read_csv(main_directory+'qp_input_output/matlab_habbit_matrix.csv', header=None).values
    print(estimate_first_cohort_habbit_matrix)
    print(max_absolute_difference_between_matrix(first_cohort_habbit_matrix, estimate_first_cohort_habbit_matrix))

    # 第一年改版矩陣的差異 (估計 vs 真實)
    estimate_first_cohort_ver_change_matrix = pd.read_csv(main_directory+'qp_input_output/matlab_version_change_matrix.csv', header=None).values
    first_cohort_ver_change_matrix = pd.read_csv(main_directory+'matrix_and_vector/first_cohort_ver_change_matrix.csv').values
    print(estimate_first_cohort_ver_change_matrix)
    print(first_cohort_ver_change_matrix)
    print(max_absolute_difference_between_matrix(estimate_first_cohort_ver_change_matrix, first_cohort_ver_change_matrix))