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
    cohort_gammafit_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_gamma_fit_mean.csv')['0']
    cohort_midpoint_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_midpoint_mean.csv')['0']
    cohort_qp_mean = pd.read_csv(main_dir+'data_for_draw_fig/cohort_qp_mean.csv')['0']

    population_gammafit_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_gamma_fit_mean.csv')['0']
    population_midpoint_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_midpoint_mean.csv')['0']
    population_qp_mean = pd.read_csv(main_dir+'data_for_draw_fig/population_qp_mean.csv')['0']
    
    # probability vector estimation(1997 cohort with 1998 questionnaire)
    estimate_first_year_cohort_vec = \
        pd.read_csv(main_dir+'qp_input_output/matlab_new_cohort_97_vec.csv', header=None)[0]
    estimate_first_year_pop_vec = \
        pd.read_csv(main_dir+'matrix_and_vector/estimate_first_yr_population_second_yr_q_vec.csv')['0']
    # Relative frequency
    estimate_first_year_pop_vec = estimate_first_year_pop_vec.div(estimate_first_year_pop_vec.sum())

    return cohort_gammafit_mean, cohort_midpoint_mean, cohort_qp_mean, \
            population_gammafit_mean, population_midpoint_mean, population_qp_mean, \
            estimate_first_year_cohort_vec, estimate_first_year_pop_vec

def read_true_values(main_dir, simulation_configs): 
    '''
    讀取正確答案
    1. 平均值
    2. Probability vectors
    '''
    true_pop_means = simulation_configs.distribution_list
    true_cohort_means = simulation_configs
    print(true_pop_means)
    print(true_pop_means[0].kwds)
    print(type(true_pop_means[0].kwds))
    # TODO: 把distribution參數從kmds當中儲存下來
    # TODO: cohort的true mean需要想想怎麼取得，之前應該有計算過
    pass

if __name__=='__main__':
    simulationConfig = Config_simul()
    # TODO: create directory for estimation error
    # TODO: 將directory的指定方式，命名方式，也加入config當中
    # read estimation results
    estimations = read_estimation_results(simulationConfig.main_directory)
    # read true values
    true_values = read_true_values(simulationConfig.main_directory, simulationConfig)
    pass