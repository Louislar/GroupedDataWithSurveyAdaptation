'''
Compute the error of estimation by the distance metrics mentioned in the article
1. Mean estimation error(Gamma fit/Midpoint/VAM)(Population/Cohort)
2. Probability vector estimation error(Linfinity/KL divergence)(Population/Cohort)
3. Bootstrap analysis 95% CI (.ipynb on healthdb)
'''

import numpy as np 
import pandas as pd 
from config import Config_simul

def read_estimation_results(main_dir): 
    '''
    讀取估計結果
    1. 平均值估計結果(Gamma fit/Midpoint/VAM)(Population/Cohort)
    2. Probability vector estimation(Population/Cohort)
    '''
    cohort_gammafit_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_gamma_fit_mean.csv')['0'].values
    cohort_midpoint_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_midpoint_mean.csv')['0'].values
    cohort_qp_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_qp_mean.csv')['0'].values

    population_gammafit_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_gamma_fit_mean.csv')['0'].values
    population_midpoint_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_midpoint_mean.csv')['0'].values
    population_qp_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_qp_mean.csv')['0'].values
    
    # probability vector estimation(1997 cohort/population with 1998 questionnaire)
    estimate_first_yr_cohort_revisioned_vec = \
        pd.read_csv(main_dir+'qp_input_output/matlab_new_cohort_97_vec.csv', header=None).values
    estimate_first_yr_pop_revisioned_vec = pd.read_csv(main_dir+'matrix_and_vector/estimate_first_yr_population_second_yr_q_vec.csv')
    estimate_first_yr_pop_revisioned_vec /= estimate_first_yr_pop_revisioned_vec.sum()
    estimate_first_yr_pop_revisioned_vec = estimate_first_yr_pop_revisioned_vec.values

    return cohort_gammafit_mean[0], cohort_midpoint_mean[0], cohort_qp_mean[0], cohort_qp_mean[1], \
            population_gammafit_mean[0], population_midpoint_mean[0], population_qp_mean[0], population_qp_mean[1], \
            estimate_first_yr_cohort_revisioned_vec, estimate_first_yr_pop_revisioned_vec

def read_true_values(main_dir, simulation_configs): 
    '''
    讀取正確答案
    1. 平均值(平均值的正確答案也可以從sample當中計算得到)(只給第一年的平均值)
    2. Probability vectors
    '''
    # Mean from the underlying distribution
    true_pop_means = [dis.kwds for dis in simulation_configs.distribution_list]
    for i in range(len(true_pop_means)):
        tmp = 1
        for k in true_pop_means[i]:
            tmp*=true_pop_means[i][k]
        true_pop_means[i] = tmp
    true_pop_means = true_pop_means[0]
    # Mean from samples
    first_yr_df = pd.read_csv(main_dir+'year_0.csv')
    second_yr_df = pd.read_csv(main_dir+'year_1.csv')
    ## population
    true_pop_sample_mean = first_yr_df['sample'].mean()
    ## cohort
    first_cohort_df = first_yr_df.loc[first_yr_df['id'].isin(second_yr_df['id']), :]
    true_cohort_sample_mean = first_cohort_df['sample'].mean()

    # Probability vectors
    ## 正確的第一年cohort改版後向量(第一年填寫第二年問卷)
    real_first_yr_cohort_revisioned_vec = pd.read_csv(main_dir+'matrix_and_vector/first_cohort_second_year_q_vec.csv')
    real_first_yr_cohort_revisioned_vec /= real_first_yr_cohort_revisioned_vec.sum()
    real_first_yr_cohort_revisioned_vec = real_first_yr_cohort_revisioned_vec.values
    ## 正確的97年population向量
    real_first_yr_population_revisioned_vec = pd.read_csv(main_dir+'matrix_and_vector/first_year_population_second_year_q_vec.csv')
    real_first_yr_population_revisioned_vec /= real_first_yr_population_revisioned_vec.sum()
    real_first_yr_population_revisioned_vec = real_first_yr_population_revisioned_vec.values

    return true_pop_means, true_pop_sample_mean, true_cohort_sample_mean, \
        real_first_yr_cohort_revisioned_vec, real_first_yr_population_revisioned_vec

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

def max_absolute_difference_between_vector(a_vector, b_vector):
    """
    計算向量間的absolute difference，每一個cell/entry分開計算，然後取最大的
    兩個向量大小要相同
    """
    a_minus_b_abs = np.abs(np.subtract(a_vector, b_vector))
    max_absolute_difference = np.amax(a_minus_b_abs)
    return max_absolute_difference

if __name__=='__main__':
    simulationConfig = Config_simul()
    # TODO: create directory for estimation error
    # TODO: 將directory的指定方式，命名方式，也加入config當中
    # read estimation results
    estimations = read_estimation_results(simulationConfig.main_directory)
    # read true values
    true_values = read_true_values(simulationConfig.main_directory, simulationConfig)
    # compute difference between estimation and true values
    ## mean of cohort
    error_cohort_gammafit_mean = abs(estimations[0] - true_values[2])
    error_cohort_midpoint_mean = abs(estimations[1] - true_values[2])
    error_cohort_qp_midpoint_mean = abs(estimations[2] - true_values[2])
    error_cohort_qp_gammafit_mean = abs(estimations[3] - true_values[2])
    ## mean of population
    error_pop_gammafit_mean = abs(estimations[4] - true_values[0])
    error_pop_midpoint_mean = abs(estimations[5] - true_values[0])
    error_pop_qp_midpoint_mean = abs(estimations[6] - true_values[0])
    error_pop_qp_gammafit_mean = abs(estimations[7] - true_values[0])

    ## probability vector
    linf_cohort_vec = max_absolute_difference_between_vector(true_values[3], estimations[8])
    kl_cohort_vec = KL_divergence_between_vector(true_values[3], estimations[8])
    linf_pop_vec = max_absolute_difference_between_vector(true_values[4], estimations[9])
    kl_pop_vec = KL_divergence_between_vector(true_values[4], estimations[9])

    output_df = pd.DataFrame({
        'error_cohort_gammafit_mean': error_cohort_gammafit_mean, 
        'error_cohort_midpoint_mean': error_cohort_midpoint_mean, 
        'error_cohort_qp_midpoint_mean': error_cohort_qp_midpoint_mean,
        'error_cohort_qp_gammafit_mean': error_cohort_qp_gammafit_mean, 
        'error_pop_gammafit_mean': error_pop_gammafit_mean, 
        'error_pop_midpoint_mean': error_pop_midpoint_mean, 
        'error_pop_qp_midpoint_mean': error_pop_qp_midpoint_mean, 
        'error_pop_qp_gammafit_mean': error_pop_qp_gammafit_mean, 
        'linf_cohort_vec': linf_cohort_vec, 
        'kl_cohort_vec': kl_cohort_vec, 
        'linf_pop_vec': linf_pop_vec, 
        'kl_pop_vec': kl_pop_vec
    }, index=['value'])
    output_df.to_csv(simulationConfig.main_directory+'estimation_error.csv')
