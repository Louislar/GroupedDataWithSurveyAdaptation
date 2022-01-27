'''
我在cdf_likelihood_study.R 已經完成了 
所以這邊會直接呼叫R那邊的code就好 
目前呼叫的版本為 "固定區間threshold，用CDF+MLE，估計母體distribution參數" 
預期要支援的母體distribution有: 
    1. weibull
    2. exponential 
    3. gamma 
    4. log normal 


理論上要做以下事情:  
計算CDF的maximumlikelihood
使用論文上的ordered probit model的公式
step 1: CDF的公式
step 2: threshold與選項的轉換函數
step 3: maximum likelihood 計算，求出發生最大值的相應參數 

但是我在R中實作好了，所以直接call R的function即可 


'''

import rpy2.robjects as ro
from rpy2.robjects.packages import importr 
import pandas as pd 

# 呼叫r當中的library
ro.r('library(nloptr)') 
ro.r('library(EnvStats)') 


def threshold_converter(parameter_list):
    """
    選項數值轉thrshold
    例如: 選項1就應該轉成threshold 1與threshold 2
    """
    pass


def extimate_distribution_by_CDF_and_MLE(Datta, expected_distribution, thresholds):
    """
    使用CDF加上MLE，估計給定的distribution的參數 

    Input: 
    :Datta:         
        資料 observation 

    :expected_distribution: 
        給定的distribution，目前預計支援四種distribution 
            1. weibull = 'weibull' 
            2. exponential = 'exp' 
            3. gamma = 'gamma' 
            4. log normal = 'lognormal' 

    :thresholds:    
        每個區間的上下界 threshold 

    Output: 
    :best_param_list: 最好的參數list 
    :max_log_likelihood: 最大log likelihood 
    :cumulative_probability_of_each_segement: 區段累積機率(最佳參數情況下) == 每個threshold區間的累積機率 

    """
    print('------estimate gamma (R)------')
    ro.globalenv['D'] = ro.FloatVector(Datta) 
    ro.globalenv['initial_threshold'] = ro.FloatVector(thresholds) 

    # 每個distribution有各自的: --> 固定變數名稱
    # 1. parameter上下界                    theta_lower_bound, theta_upper_bound
    # 2. parameter inital value             initial_theta
    # 3. cdf likelihood 計算 function       ordered_probit_fix_threshold
    
    if expected_distribution == 'gamma': 
        # 1. define the distribution's parameters lower bound and upper bound 
        ro.r('theta_lower_bound <- c(10^-20, 10^-20)') 
        ro.r('theta_upper_bound <- c(Inf, Inf)') 

        # 2. initial value for estimation (moment estimator) 
        ro.r('momalpha <- mean(D)^2/var(D) \n\
            mombeta <- var(D)/mean(D)') 
        ro.r('initial_theta <- c(momalpha, mombeta)')

        # 3. define cdf likelihood calculator function 
        ro.r('ordered_probit_fix_threshold <- function(theta, datta, threshold_vec) \n\
                {\n\
                alpha <- theta[1]; g_theta <- theta[2]; \n\
                threshold_vec <- c(0, threshold_vec, Inf)  \n\
                data_threshold_1 <- threshold_vec[datta] \n\
                data_threshold_2 <- threshold_vec[datta+1] \n\
                total_likelihood = 0 \n\
                for(i in 1:length(data_threshold_1)) \n\
                { \n\
                    tmp_likelihood <- max(10^-20, pgamma(data_threshold_2[i], shape = alpha, scale = g_theta) - pgamma(data_threshold_1[i], shape = alpha, scale = g_theta)) \n\
                    total_likelihood = total_likelihood + log(tmp_likelihood) \n\
                    if(total_likelihood <= -10^10) \n\
                    { \n\
                    print("BBBBBBBBBBBBBOOOOOOOOOOMMMMMMM") \n\
                    } \n\
                } \n\
                print(total_likelihood) \n\
                total_likelihood * (-1)}') 

    elif expected_distribution == 'exp': 
        # 1. define the distribution's parameters lower bound and upper bound 
        ro.r('theta_lower_bound <- c(10^-20)') 
        ro.r('theta_upper_bound <- c(Inf)') 

        # 2. initial value for estimation (moment estimator) 
        ro.r('momalpha <- mean(D)') 
        ro.r('initial_theta <- c(momalpha)') 

        # 3. define cdf likelihood calculator function 
        ro.r('ordered_probit_fix_threshold <- function(theta, datta, threshold_vec) \n\
                {\n\
                alpha <- theta[1] \n\
                threshold_vec <- c(0, threshold_vec, Inf)  \n\
                data_threshold_1 <- threshold_vec[datta] \n\
                data_threshold_2 <- threshold_vec[datta+1] \n\
                total_likelihood = 0 \n\
                for(i in 1:length(data_threshold_1)) \n\
                { \n\
                    tmp_likelihood <- max(10^-20, pexp(data_threshold_2[i], rate = alpha) - pexp(data_threshold_1[i], rate = alpha)) \n\
                    total_likelihood = total_likelihood + log(tmp_likelihood) \n\
                    if(total_likelihood <= -10^10) \n\
                    { \n\
                    print("BBBBBBBBBBBBBOOOOOOOOOOMMMMMMM") \n\
                    } \n\
                } \n\
                print(total_likelihood) \n\
                total_likelihood * (-1)}') 
        
    elif expected_distribution == 'weibull': 
        # 1. define the distribution's parameters lower bound and upper bound 
        ro.r('theta_lower_bound <- c(10^-20, 10^-20)') 
        ro.r('theta_upper_bound <- c(Inf, Inf)') 

        # 2. initial value for estimation (moment estimator) 
        # ref: http://www.m-hikari.com/ams/ams-2014/ams-81-84-2014/sedliackovaAMS81-84-2014.pdf
        # ref: http://search.r-project.org/library/EnvStats/html/eweibull.html 
        ro.r('momalpha_and_theta <- EnvStats::eweibull(D, method="mme")') 
        ro.r('momshape <- momalpha_and_theta$parameters[[\'shape\']]') 
        ro.r('momscale <- momalpha_and_theta$parameters[[\'scale\']]') 
        ro.r('initial_theta <- c(momshape, momscale)')


        # 3. define cdf likelihood calculator function 
        ro.r('ordered_probit_fix_threshold <- function(theta, datta, threshold_vec) \n\
                {\n\
                alpha <- theta[1]; g_theta <- theta[2]; \n\
                threshold_vec <- c(0, threshold_vec, Inf)  \n\
                data_threshold_1 <- threshold_vec[datta] \n\
                data_threshold_2 <- threshold_vec[datta+1] \n\
                total_likelihood = 0 \n\
                for(i in 1:length(data_threshold_1)) \n\
                { \n\
                    tmp_likelihood <- max(10^-20, pweibull(data_threshold_2[i], shape = alpha, scale = g_theta) - pweibull(data_threshold_1[i], shape = alpha, scale = g_theta)) \n\
                    total_likelihood = total_likelihood + log(tmp_likelihood) \n\
                    if(total_likelihood <= -10^10) \n\
                    { \n\
                    print("BBBBBBBBBBBBBOOOOOOOOOOMMMMMMM") \n\
                    } \n\
                } \n\
                print(total_likelihood) \n\
                total_likelihood * (-1)}') 

    elif expected_distribution == 'lognormal': 
        # 1. define the distribution's parameters lower bound and upper bound 
        ro.r('theta_lower_bound <- c(-Inf, 10^-20)') 
        ro.r('theta_upper_bound <- c(Inf, Inf)') 

        # 2. initial value for estimation (moment estimator) 
        # ref: https://www.real-statistics.com/distribution-fitting/method-of-moments/method-of-moments-lognormal-distribution/
        ro.r('mommean <- log(mean(D) / sqrt(var(D)/(mean(D)^2) + 1)) \n\
            momsd <- sqrt(log(var(D)/(mean(D)^2) + 1))') 
        ro.r('initial_theta <- c(mommean, momsd)') 


        # 3. define cdf likelihood calculator function 
        ro.r('ordered_probit_fix_threshold <- function(theta, datta, threshold_vec) \n\
                {\n\
                alpha <- theta[1]; g_theta <- theta[2]; \n\
                threshold_vec <- c(0, threshold_vec, Inf)  \n\
                data_threshold_1 <- threshold_vec[datta] \n\
                data_threshold_2 <- threshold_vec[datta+1] \n\
                total_likelihood = 0 \n\
                for(i in 1:length(data_threshold_1)) \n\
                { \n\
                    tmp_likelihood <- max(10^-20, plnorm(data_threshold_2[i], meanlog = alpha, sdlog = g_theta) - plnorm(data_threshold_1[i], meanlog = alpha, sdlog = g_theta)) \n\
                    total_likelihood = total_likelihood + log(tmp_likelihood) \n\
                    if(total_likelihood <= -10^10) \n\
                    { \n\
                    print("BBBBBBBBBBBBBOOOOOOOOOOMMMMMMM") \n\
                    } \n\
                } \n\
                print(total_likelihood) \n\
                total_likelihood * (-1)}') 
        pass


    # 尋找數值解 
    ro.r('library(nloptr) \n\
            opt_list <- list( \n\
            "xtol_rel" = 1e-10 \n\
            )') 
    ro.r('maxll_result <- nloptr::bobyqa(fn = ordered_probit_fix_threshold, x0 = initial_theta, \
        datta=D , threshold_vec=initial_threshold, lower = theta_lower_bound, \
            upper = theta_upper_bound, control = opt_list) \n\
            print(maxll_result)') 
    
    # 回傳: 
    # 1. 估計的參數 (list) 
    # 2. 最佳參數的log likelihood = max log likelihood 
    # 3. 最佳參數的區段累積機率 
    ro.r('best_param_vec <- maxll_result$par') 
    ro.r('max_log_likelihood <- (-1) * maxll_result$value') 
    # 計算區段累積機率  
    ro.r('initial_threshold <- c(0, initial_threshold, Inf)') 
    ro.r('cumu_prob_of_each_seg <- c()')
    if expected_distribution == 'gamma': 
        ro.r('for(i in 1:length(initial_threshold)-1) \n\
            { \n\
                #print(i) \n\
                cumu_prob_of_each_seg[i] <- \n\
                pgamma(initial_threshold[i+1], shape = best_param_vec[1], scale = best_param_vec[2]) - \n\
                pgamma(initial_threshold[i], shape = best_param_vec[1], scale = best_param_vec[2]) \n\
            }')
    elif expected_distribution == 'exp': 
        ro.r('for(i in 1:length(initial_threshold)-1) \n\
            { \n\
                #print(i) \n\
                cumu_prob_of_each_seg[i] <- \n\
                pexp(initial_threshold[i+1], rate = best_param_vec[1]) - \n\
                pexp(initial_threshold[i], rate = best_param_vec[1]) \n\
            }')
    elif expected_distribution == 'weibull': 
        ro.r('for(i in 1:length(initial_threshold)-1) \n\
            { \n\
                #print(i) \n\
                cumu_prob_of_each_seg[i] <- \n\
                pweibull(initial_threshold[i+1], shape = best_param_vec[1], scale = best_param_vec[2]) - \n\
                pweibull(initial_threshold[i], shape = best_param_vec[1], scale = best_param_vec[2]) \n\
            }')
    elif expected_distribution == 'lognormal': 
        ro.r('for(i in 1:length(initial_threshold)-1) \n\
            { \n\
                #print(i) \n\
                cumu_prob_of_each_seg[i] <- \n\
                plnorm(initial_threshold[i+1], meanlog = best_param_vec[1], sdlog = best_param_vec[2]) - \n\
                plnorm(initial_threshold[i], meanlog = best_param_vec[1], sdlog = best_param_vec[2]) \n\
            }')
        
    ro.r('')

    best_param_list = ro.globalenv['best_param_vec'] 
    max_log_likelihood = ro.globalenv['max_log_likelihood'] 
    cumu_prob_of_each_threshold = ro.globalenv['cumu_prob_of_each_seg'] 

    return list(best_param_list), list(max_log_likelihood)[0], list(cumu_prob_of_each_threshold) 

if __name__ == "__main__": 
    tmp_df = pd.read_csv('simulation_q_result.csv') 
    tmp_1997_data = tmp_df['1997']
    # print(tmp_df.head(20)) 
    # print(tmp_1997_data) 
    param_list, max_log_likelihood, cumu_prob_list = extimate_distribution_by_CDF_and_MLE(tmp_1997_data, 'gamma', [1, 2, 3]) 
    print('best parameter list: ', param_list) 
    print('maximum log likelihood: ', max_log_likelihood) 
    print('cumulative probability of each threshold interval: ', cumu_prob_list)
