#!/usr/bin/env python
# coding: utf-8

# ## Gamma參數距離要差多大，QP結果才會在95%裡面
# - 輸出成表格，方便觀察有無掉入95%的信心水準內

# In[1]:


import pandas as pd 
import numpy as np 
import scipy.stats as st 
import sys 
sys.path.append('./parallel_resampling/')
import gamma_generater
from collections import Counter 
import matplotlib.pyplot as plt 
import time


# In[2]:


# Parameters 
sample_size = 20000    # 樣本大小
sample_count = 1000    # 取樣次數
# sample_size = 100    # 樣本大小
# sample_count = 10    # 取樣次數
gamma_param1 = [1, 3]
gamma_param2 = [1, 3]
gamma_shift_step = 1e-2    # 單次參數調整大小
gamma_shift_count = 100    # 參數調整總次數 (unused)
target_linf = 0.01392    # 目標的 L infinity大小 (unused)


# In[3]:


def max_absolute_difference_between_vector(a_vector, b_vector):
    """
    計算向量間的absolute difference，每一個cell/entry分開計算，然後取最大的
    兩個向量大小要相同
    """
    a_minus_b_abs = np.abs(np.subtract(a_vector, b_vector))
    max_absolute_difference = np.amax(a_minus_b_abs)
    return max_absolute_difference


# In[4]:


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


# In[5]:


def sample_2_gamma(
    gamma_param, sample_s, sample_c, rand_seed_start
):
    cur_rand_seed = rand_seed_start
    gamma_samples_list = []
    
    for i in range(sample_c): 
        np.random.seed(cur_rand_seed) 
        gamma_samples_list.append(
            st.gamma.rvs(a=gamma_param[0], scale=gamma_param[1], size=sample_s) 
        )
        cur_rand_seed += 1
    gamma_samples1 = np.vstack(gamma_samples_list)
    
    return gamma_samples1


# In[6]:


def gamma_samples_2_p_vec(samples): 
#     print(samples)
    start_time = time.time()
    a = np.apply_along_axis(
        lambda x: gamma_generater.fill_questionnaire(
            gamma_generater.questionnaire_1998, 
            x, 
            print_log=False, 
            return_list=True
        ), 
        axis=1, 
        arr=samples
    )
    tmp_time1=time.time()
    print('fill questionnaire time: ', tmp_time1 - start_time)
    print(a)
    b = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(gamma_generater.questionnaire_1998)+1), 
        axis=1, 
        arr=a
    )
    b = b[:, 1:]
    tmp_time2 = time.time()
    print('bincount time: ', tmp_time2 - start_time)
    print(b)
    c = np.apply_along_axis(
        lambda x: x/x.sum(), 
        axis=1, 
        arr=b
    )
    tmp_time3 = time.time()
    print('to precentage time: ', tmp_time3 - start_time)
    print(c)
    return c


# In[7]:


# calculate lininity norm 
def pvec_2_linf(gamma1_p_vec, gamma2_p_vec): 
    '''
    :gamma1_p_vec: (np.array) 2D array, last axis is a sample 
    '''
    x, y = np.meshgrid(range(gamma1_p_vec.shape[0]), range(gamma2_p_vec.shape[0]))
    x=x.flatten()
    y=y.flatten()
    p_vec1 = gamma1_p_vec[x, :]
    p_vec2 = gamma2_p_vec[y, :]
    after_abs_subtract = np.abs(np.subtract(p_vec1, p_vec2))
    linf_norm = np.amax(after_abs_subtract, axis=1)
    print('Shape of L infinity: ', linf_norm.shape)
    return linf_norm


# In[8]:


# calculate KL divergence 
def pvec_2_kl(gamma1_p_vec, gamma2_p_vec): 
    x, y = np.meshgrid(range(gamma1_p_vec.shape[0]), range(gamma2_p_vec.shape[0]))
    x=x.flatten()
    y=y.flatten()
    p_vec1 = gamma1_p_vec[x, :]
    p_vec2 = gamma2_p_vec[y, :]
    after_log = np.log(np.divide(p_vec1, p_vec2))
    after_log = np.where(np.isneginf(after_log), 0, after_log)  # 處理a column中有0的情況，要回傳0，而不是nan
    after_log = np.multiply(p_vec1, after_log)
    expect_val = np.sum(after_log, axis=1)
    return expect_val 


# In[9]:


# Input: 兩個gamma的參數, sample大小, sample次數
# Output: sample間各種Linf的出現頻率, 平均Linf, 5%和95% quantile的Linf

def two_gamma_Linf(gamma_param1, gamma_param2, sample_s, sample_c, rnd_seed_start): 
    # 1. sample 2 Gamma 
    samples1 = sample_2_gamma(
        gamma_param1, sample_size, sample_count, rand_seed_start=rnd_seed_start
    )
    samples2 = sample_2_gamma(
        gamma_param2, sample_size, sample_count, rand_seed_start=rnd_seed_start+sample_count
    )
    # 2. Convert samples to probability vector 
    pVec1 = gamma_samples_2_p_vec(samples1)
    pVec2 = gamma_samples_2_p_vec(samples2)
    # 3. L inf between 2 vector 
    Linfs = pvec_2_linf(pVec1, pVec2)
    # 3.1 KL divergence between 2 vector 
    KLs = pvec_2_kl(pVec1, pVec2)
    # 4. 5% and 95% quantile
    q5 = np.quantile(Linfs, 0.05)
    q95 = np.quantile(Linfs, 0.95)
    q2_5 = np.quantile(Linfs, 0.025)
    q97_5 = np.quantile(Linfs, 0.975)
    # 4.1 KL quantiles 
    q5_kl = np.quantile(KLs, 0.05)
    q95_kl = np.quantile(KLs, 0.95)
    q2_5_kl = np.quantile(KLs, 0.025)
    q97_5_kl = np.quantile(KLs, 0.975)
    # 5. basic statistics 
    basic_stats = st.describe(Linfs)
    basic_stats_kl = st.describe(KLs)
    
    return q5, q95, q2_5, q97_5, basic_stats, Linfs,            q5_kl, q95_kl, q2_5_kl, q97_5_kl, basic_stats_kl, KLs





# ## Parallel sampling and questionnaire filling test

# In[10]:


def _func1(a): 
    return a<1
def _func2(a): 
    return (a>=1 and a<2.5)
def _func3(a): 
    return (a>=2.5 and a<4.5)
def _func4(a): 
    return (a>=4.5 and a<6.5)
def _func5(a): 
    return a>=6.5
class fill_qer(object): 
    def __init__(self, print_log, return_list):
        self.qer = [
            _func1, 
            _func2, 
            _func3, 
            _func4, 
            _func5
        ]
        self.print_log = print_log
        self.return_list = return_list
    def fill_questionnaire(self, questionnaire, answer, print_log=True, return_list=False): 
        '''
        讓多筆資料填寫設計好的問卷，以美兆資料集問卷做為參考
        questionnaire: 問卷題目，題目中選項的條件式
        answer: 填答的資料，通常是用某個distribution sample出來的資料，連續數值型資料
        ref: https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
        '''
        choice_pick_list = []   # 選項統計
        for one_ans in answer: 
            after_fill_in_list = [choice_func(one_ans) for choice_func in questionnaire]
            if print_log == True: 
                print(after_fill_in_list)
                print(one_ans)
            choice_idx = [i for i, x in enumerate(after_fill_in_list) if x] 
            choice_pick_list.append(choice_idx[0]+1)

        if return_list == True: 
            return choice_pick_list
        # print(choice_pick_list)
        choice_dict = dict(Counter(choice_pick_list))
        return choice_dict
    def __call__(self, x):
        return self.fill_questionnaire(
            self.qer, 
            x, 
            print_log=self.print_log, 
            return_list=self.return_list
        )


# In[11]:


class np_bincounter(object): 
    def __init__(self, length): 
        self.length=length
    def __call__(self, x): 
        return np.bincount(x, minlength=self.length)
class np_pvec_maker(object): 
    def __init__(self): 
        pass
    def __call__(self, x): 
        return x/x.sum()


# In[12]:


def parallel_gamma_samples_2_p_vec(samples): 
#     print(samples)
    start_time = time.time()
    a = parallel_apply_along_axis(
        fill_qer(
            print_log=False, 
            return_list=True
        ), 
        axis=1, 
        arr=samples
    )
    tmp_time1=time.time()
#     print('fill questionnaire time: ', tmp_time1 - start_time)
#     print(a)
    b = parallel_apply_along_axis(
        np_bincounter(6), 
        axis=1, 
        arr=a
    )
    b = b[:, 1:]
    tmp_time2 = time.time()
#     print('bincount time: ', tmp_time2 - start_time)
#     print(b)
    c = parallel_apply_along_axis(
        np_pvec_maker(), 
        axis=1, 
        arr=b
    )
    tmp_time3 = time.time()
#     print('to precentage time: ', tmp_time3 - start_time)
#     print(c)
    return c


# In[13]:


'''
Parallelize the numpy function apply_along_axis()
ref: https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis 
ref: https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function 
ref: https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror
'''
import multiprocessing

import numpy as np

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count()//2+1)]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)
def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


# In[14]:


# parallel sampling test
# _samples = sample_2_gamma(gamma_param1, sample_s=57000, sample_c=400, rand_seed_start=0)
# pVec1 = gamma_samples_2_p_vec(_samples)
# print(_samples.shape)
# pVec1


# In[15]:


# _samples = sample_2_gamma(gamma_param1, sample_s=57000, sample_c=400, rand_seed_start=0)
# pVec1 = parallel_gamma_samples_2_p_vec(_samples)
# print(_samples.shape)
# pVec1


# In[16]:


# Input: 兩個gamma的參數, sample大小, sample次數
# Output: sample間各種Linf的出現頻率, 平均Linf, 5%和95% quantile的Linf

def two_gamma_Linf(gamma_param1, gamma_param2, sample_s, sample_c, rnd_seed_start): 
    # 1. sample 2 Gamma 
    samples1 = sample_2_gamma(
        gamma_param1, sample_size, sample_count, rand_seed_start=rnd_seed_start
    )
    samples2 = sample_2_gamma(
        gamma_param2, sample_size, sample_count, rand_seed_start=rnd_seed_start+sample_count
    )
    # 2. Convert samples to probability vector 
    pVec1 = parallel_gamma_samples_2_p_vec(samples1)
    pVec2 = parallel_gamma_samples_2_p_vec(samples2)
    # 3. L inf between 2 vector 
    Linfs = pvec_2_linf(pVec1, pVec2)
    # 3.1 KL divergence between 2 vector 
    KLs = pvec_2_kl(pVec1, pVec2)
    # 4. 5% and 95% quantile
    q5 = np.quantile(Linfs, 0.05)
    q95 = np.quantile(Linfs, 0.95)
    q2_5 = np.quantile(Linfs, 0.025)
    q97_5 = np.quantile(Linfs, 0.975)
    # 4.1 KL quantiles 
    q5_kl = np.quantile(KLs, 0.05)
    q95_kl = np.quantile(KLs, 0.95)
    q2_5_kl = np.quantile(KLs, 0.025)
    q97_5_kl = np.quantile(KLs, 0.975)
    # 5. basic statistics 
    basic_stats = st.describe(Linfs)
    basic_stats_kl = st.describe(KLs)
    
    return q5, q95, q2_5, q97_5, basic_stats, Linfs,            q5_kl, q95_kl, q2_5_kl, q97_5_kl, basic_stats_kl, KLs


# In[17]:


# 根據設定的step，不斷改變gamma的參數，觀察Quantile與mean的變化
# 因為計算很久的關係，可能要先將模擬結果儲存下來
# 目前先調整第二個gamma的第二個參數
# --> 從3到5，並且step為0.01，所以總共執行200次
gamma_params_arr = np.arange(3, 5, gamma_shift_step)
gamma_shift_df = pd.DataFrame()
iter_count=0
for i in gamma_params_arr: 
    q5, q95, q2_5, q97_5, basic_stats, Linfs,    q5_kl, q95_kl, q2_5_kl, q97_5_kl, basic_stats_kl, KLs=     two_gamma_Linf(
        gamma_param1, 
        [gamma_param2[0], i], 
        sample_size, 
        sample_count, 
        iter_count
    )
    gamma_shift_df[i] = [q5, q95, q2_5, q97_5, basic_stats, q5_kl, q95_kl, q2_5_kl, q97_5_kl, basic_stats_kl]
    iter_count += sample_count
print(gamma_shift_df.iloc[:, :2])
gamma_shift_df.to_csv('./gamma_shift_inc_20000.csv')

