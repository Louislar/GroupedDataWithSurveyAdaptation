import pandas as pd 
import numpy as np
from common_module_pkg.call_r_test import estimate_gamma_distribution_by_r
from common_module_pkg import gamma_generater 
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import Counter

def find_gamma_upper_bound(gamma_pdf, initial_finding_x=0, upper_bound_threshold=10**(-2), is_print_iter=False): 
    '''
    向更大的x方向不斷尋找小於thrshold的資料點，該點即為gamma的upper bound
    **需要說明gamma distribution的pdf到後來會是遞減函數

    Input: 
    :gamma_pdf: 被尋找的distribution的pdf
    :initial_finding_x: 開始尋找的隨機變數x起點
    :upper_bound_threshold: 上限的機率閾值

    Output: 
    :upper_bound_out: 找到最小的x且最接近threshold
    '''
    print('------distribution upper bound finding------')
    # 從開始尋找的起點不斷往上加一個微小的數值，
    # 並且每次都將x加上一個微小數值，再帶入給定的pdf當中，
    # 當pdf回傳的機率值小於threshold就停止搜尋，並回傳當下的x
    
    p_from_pdf = 100000
    x = initial_finding_x
    dx = 10**(-2)   # 每次x增加的微小距離


    x_iter = []
    p_from_pdf_iter = []
    while p_from_pdf >= upper_bound_threshold: 
        x += dx
        p_from_pdf = gamma_pdf(x)
        x_iter.append(x)
        p_from_pdf_iter.append(p_from_pdf)
    
    if is_print_iter == True: 
        print('initial finding x: ', initial_finding_x)
        print('dx: ', dx)
        print('threshold: ', upper_bound_threshold)
        for x_p_pair in zip(*[x_iter, p_from_pdf_iter]): 
            print('(x, p(x)): ', x_p_pair)
        

    return x

def rejection_sampling(origin_pdf, lower_bound, upper_bound, num_of_sample, is_draw=False): 
    '''
    給pdf公式，並給定欲限制的上下限，
    利用給定的上下限框出比pdf還大的uniform distribution，
    不斷做sampling，最後得到足夠的樣本數才停止

    Input: 
    :origin_pdf: 
    :lower_bound: 
    :upper_bound: 
    :num_of_sample: 

    Output: 
    根據number_of_sample的回傳同等數量的樣本資料點

    ref: 
    np.uniform: https://ithelp.ithome.com.tw/articles/10214963
    '''
    
    print('------rejection sampling------')
    print('upper bound: ', upper_bound)
    print('lower bound: ', lower_bound)
    # 先產生upper bound與lower bound之間的數值(密度最好要能夠到像連續型)
    # 先產生準確到小數點後兩位的資料點
    x = np.linspace(lower_bound, upper_bound, num=int((upper_bound-lower_bound)/0.01) )
    print(x)
    # 求出最大p/q的比率k，(q先使用uniform distribution)
    # k*q就會能夠包住p
    def q(x): 
        return st.uniform.pdf(x, loc=lower_bound, scale=upper_bound - lower_bound)

    k = max(origin_pdf(x)/q(x))

    # 開始generate samples，iteration先設10000次
    samples = []
    samples_y = []
    reject_samples = []
    reject_samples_y = []

    iter = 100000
    for i in range(iter):  
        z = st.uniform.rvs(loc=lower_bound, scale=upper_bound - lower_bound, size=1)
        q_z = q(z)
        u_upper_bound = k*q_z
        u = st.uniform.rvs(loc=0, scale=u_upper_bound)
        # print('upper bound of u: ', u_upper_bound)

        if u <= origin_pdf(z): 
            samples.append(z[0])
            samples_y.append(u)
        else: 
            reject_samples.append(z)
            reject_samples_y.append(u)
        
        # 終止條件
        if len(samples) >= num_of_sample: 
            break
        
        

    if is_draw == True: 
        print(k)

        plt.figure()
        plt.plot(x, origin_pdf(x), 'm-')    # 紫色線
        plt.plot(x, q(x), 'r-') # 紅色線
        plt.plot(x, k*q(x), 'y--')  # 土黃色虛線
        plt.plot(samples, samples_y, 'co')
        plt.plot(reject_samples, reject_samples_y, 'k,')
        # plt.show()

    return samples

    


def estimate_distribution_iteratively(datta, tail_threshold, is_return_tail_sol=False): 
    '''
    給定sample出的資料，假設它在最後的一個選項有tail problem
    利用迭代的方式估計出最接近的distribution
    Input: 
    :datta: 原始樣本資料
    :tail_threshold: 末尾問題發生的數值 (大於等於此threshold就判斷為末尾問題的資料點)
    :is_return_tail_sol: 是否直接回傳尾段重新建設後的所有數值
    '''
    # step 1: 使用R近似出gamma distribution的參數
    # step 2: 將末尾選項用近似出的distribution分配，得到新的離散資料
    # step 3: 再重新使用新的離散資料進行step1, step2
    # 停止條件: 末尾選項與新近似的distribution差異過小 
    #           (alpha與beta都與前次iter差異小於0.00001就停止)

    gamma_param_per_iter = []
    previous_alpha = 10**5
    previous_beta = 10**5

    for i in range(100): 

        # ============step 1=============
        # 對資料做近似
        alpha_h, theta_h = estimate_gamma_distribution_by_r(datta)
        gamma_param_per_iter.append([alpha_h, theta_h])
        number_of_tail_problem_samples = len([d for d in datta if d>=tail_threshold])
        new_data = [d for d in datta if d<tail_threshold]

        print('alpha_h: ', alpha_h)
        print('beta_h: ', theta_h)

        # ============step 2=============
        # 目標: 只取gamma末尾部分的機率值作為原始資料的末尾選項的分配依據，
        # 也就是將末尾部分的樣本數量，依照末尾之後的gamma機率做分配
        # 先找到估計出來的gamma distribution的上界(p(x)最接近0的x)
        estimate_gamma_upper_bound = find_gamma_upper_bound(
            lambda x: st.gamma.pdf(x, a=alpha_h, loc=0, scale=theta_h), 
            initial_finding_x=alpha_h*theta_h, 
            upper_bound_threshold=10**(-5), 
            is_print_iter=True
        )
        # 利用upper bound進行尾段sample的重新分配
        tail_problem_resample_samples = rejection_sampling(
            lambda x: st.gamma.pdf(x, a=alpha_h, loc=0, scale=theta_h), 
            lower_bound=tail_threshold, 
            upper_bound=estimate_gamma_upper_bound, 
            num_of_sample=number_of_tail_problem_samples, 
            is_draw=False
        )
        

        # 把重新分配過的末尾選項資料點回填到資料集當中
        new_data += tail_problem_resample_samples
        # print('origin data length: ', len(datta))
        # print('new data length: ', len(new_data))
        # print(type(datta))
        # print(type(new_data))
        # print(type(tail_problem_resample_samples))
        # print(type(tail_problem_resample_samples[0]))
        # print(tail_problem_resample_samples[0])

        # 畫出來確認sample的對不對
        #       目前做出來的結果發現大於等於三的選項太過於分散(每個x都只分配到1個資料點)，
        #       原因是目前使用的x會到小數點後好幾位，所以必須要限縮到小數點後2位會比較好
        # 重新分配末尾選項後的資料集要取到小數點後第二位
        new_data = [round(a, 2) for a in new_data]

        datta_count_dict = dict(Counter(datta))
        new_data_count_dict = dict(Counter(new_data))
        # print('--------origin data dict--------')
        # print(datta_count_dict)
        # print('--------new data dict--------')
        # print(new_data_count_dict)

        # 疊gamma的曲線圖
        # x = np.linspace(0, 14, num=int((14-0)/0.01) )
        # gamma_x = st.gamma.pdf(x, a=alpha_h, loc=0, scale=beta_h)

        # ax1 = gamma_generater.questionnaire_result_bar(datta_count_dict.keys(), datta_count_dict.values(), True)
        # ax1.plot(x, gamma_x * len(new_data), 'y--')
        # ax2 = gamma_generater.questionnaire_result_bar(new_data_count_dict.keys(), new_data_count_dict.values(), another_fig=True, barlabel=False)
        # ax2.plot(x, gamma_x * len(new_data), 'y--')

        # 只畫末尾部分的圖
        # tail_x = np.linspace(tail_threshold, estimate_gamma_upper_bound, num=int((estimate_gamma_upper_bound-tail_threshold)/0.01))
        # tail_gamma_x = st.gamma.pdf(tail_x, a=alpha_h, loc=0, scale=beta_h)
        # tail_data_count_dict = {k: new_data_count_dict[k] for k in new_data_count_dict if k>=3}

        # plt.figure()
        # ax = plt.subplot()
        # ax.plot(tail_x, tail_gamma_x*sum(tail_data_count_dict.values()), 'y--')
        # ax.bar(tail_data_count_dict.keys(), tail_data_count_dict.values(), width=0.3) 

        # plt.show() 

        # 更新下一個iter的input
        datta = new_data
        # 確認近似的gamma distribution的參數與現在估計的是不是很接近了
        if abs(alpha_h-previous_alpha) < 10**(-3) and abs(theta_h-previous_beta) < 10**(-3): 
            print('iter stop')
            break
        # 更新alpha beta
        previous_alpha = alpha_h
        previous_beta = theta_h

    
    # 畫出每次iter近似出的gamma
    plt.figure()
    ax = plt.subplot()
    iter_count = 1
    x = np.linspace(0, 14, num=int((14-0)/0.01) )
    for a_b_list in gamma_param_per_iter: 
        gamma_x = st.gamma.pdf(x, a=a_b_list[0], loc=0, scale=a_b_list[1])
        ax.plot(x, gamma_x, label= 'iter: ' + str(iter_count))
        iter_count += 1
    # 原始近似結果，與直接拿掉末尾結果
    gamma_x = st.gamma.pdf(x, a=1.9536674, loc=0, scale=0.6791141)
    ax.plot(x, gamma_x, '--', label= 'origin')

    gamma_x = st.gamma.pdf(x, a=2.5183706   , loc=0, scale=0.4011609)
    ax.plot(x, gamma_x, '--', label= 'origin(no tail)')

    ax.legend(loc='upper right')
    plt.show()

    if is_return_tail_sol == True: 
        return datta, gamma_param_per_iter

    return gamma_param_per_iter

if __name__ == "__main__":
    # estimate_distribution_iteratively([1, 2, 3, 4, 5], 5)

    # gamma distribution 尋找最接近threshold的x位置 (測試用)
    # find_gamma_upper_bound(
    #     lambda x: st.gamma.pdf(x, a=2.5183706, scale=0.4011609, loc=0), 
    #     0, 
    #     10**(-5)
    # )


    # rejection sampling (測試用)
    # rejection_sampling(
    #     lambda x: st.gamma.pdf(x, a=2.5183706, scale=0.4011609, loc=0), 
    #     lower_bound=0, 
    #     upper_bound=50, 
    #     num_of_sample=300
    # )

    # (測試用) 跑一次iter
    with_LTPA_after_screening_df = pd.read_csv('../data/with_LTPA_after_screening.csv')
    duration_1997_df = with_LTPA_after_screening_df.loc[with_LTPA_after_screening_df['yr'] == 1997, 'duration']
    duration_1997_list = list(duration_1997_df)
    print('data length: ', len(duration_1997_list))
    
    # plt.figure()
    # plt.bar(duration_1997_dict.keys(), duration_1997_dict.values())
    # plt.show()

    # iteratively estimate
    estimate_distribution_iteratively(duration_1997_list, 3)
