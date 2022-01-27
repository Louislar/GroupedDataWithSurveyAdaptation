'''
step1: 生成出gamma distribution
step2: 生成出填問卷的函數(多種問卷)，以及填完問卷的資料統計、繪圖
step3: 使用生成的gamma distribution填寫問卷，並產生結果
step4: 問卷填寫完的結果，再給予每一個選項一個自定義的數值(e.g. duration)
'''

import math
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd 

questionnaire_value_1997 = [0.5, 1.5, 2.5, 3.0]
questionnaire_value_1998 = [0.5, 1.75, 3.5, 5.5, 6.5]
questionnaire_value_2009 = [0.0, 0.125, 0.5, 1.125, 1.625, 1.875, 2.05, 3.15, 4.2875, 4.8125, 5.425, 8.05, \
                            11.8125, 13.5625, 20.125, 30.625, 35.0]

questionnaire_1997 = [
    lambda a: a<1, 
    lambda a: a>=1 and a<2, 
    lambda a: a>=2 and a<3, 
    lambda a: a>=3
]

questionnaire_1998 = [
    lambda a: a<1, 
    lambda a: a>=1 and a<2.5, 
    lambda a: a>=2.5 and a<4.5, 
    lambda a: a>=4.5 and a<6.5, 
    lambda a: a>=6.5
]

questionnaire_2009 = [
    lambda a: a<=0, 
    lambda a: a>0 and a<=0.25, 
    lambda a: a>0.25 and a<=0.75, 
    lambda a: a>0.75 and a<=1.5, 
    lambda a: a>1.5 and a<=1.75, 
    lambda a: a>1.75 and a<=2, 
    lambda a: a>2 and a<=2.1, 
    lambda a: a>2.1 and a<=4.2, 
    lambda a: a>4.2 and a<=4.375, 
    lambda a: a>4.375 and a<=5.25, 
    lambda a: a>5.25 and a<=5.6, 
    lambda a: a>5.6 and a<=10.5, 
    lambda a: a>10.5 and a<=13.125, 
    lambda a: a>13.125 and a<=14, 
    lambda a: a>14 and a<=26.25, 
    lambda a: a>26.25 and a<= 35, 
    lambda a: a>35
]

def gamma_pdf_convert(x, alpha, beta, location=-1): 
    '''
    輸入隨機變數x，以及gamma distribution的參數a, b，
    回傳x對應的機率
    '''
    return gamma.pdf(x, a=alpha, loc=location, scale=1/beta)

def gamma_sampling(num, alpha, beta, location=-1): 
    '''
    給定gamma distribution的參數alpha與beta，
    再從中sample出指定數量(num)的資料點
    '''
    return gamma.rvs(size=num, a=alpha, loc=location, scale=1/beta)

def darw_pdf(x, y): 
    '''
    畫出PDF
    '''
    plt.figure()
    ax = plt.subplot()
    ax.plot(x, y)
    plt.show()

def draw_sampling_points_hist(sampling_points): 
    '''
    將從distribution隨機取樣的資料點，使用長條圖畫出
    '''
    plt.figure()
    ax = plt.subplot()
    ax.hist(sampling_points, density=True, histtype='stepfilled', alpha=0.5)
    # plt.show()

def questionnaire_result_bar(x, y, another_fig=True, barlabel=True): 
    '''
    問卷填答結果繪圖，使用長條圖畫出
    因為已知每個長條圖的x與y，所以使用bar chart，而不用hist
    ref: https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    '''
    def autolabel(rects):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    if another_fig == True: 
        plt.figure()
    ax = plt.subplot()
    rects = ax.bar(x, y)
    if barlabel == True: 
        autolabel(rects)
    return ax
    # plt.show()

def fill_questionnaire(questionnaire, answer, print_log=True, return_list=False): 
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

def change_questionnaire_choice_to_value(choice_dict, value_list): 
    '''
    obsolete
    把選項的編號轉換成實際的數值
    輸入選項選擇結果(choice_dict)，以及各選項對應的數值(value_list)
    回傳選項選擇結果字典檔的key值為對應數值
    '''
    new_dict = {k:0 for k in choice_dict}
    return new_dict

def distribution_evaluation(distribution_pdf_func):
    '''
    評估估計出來的分配與原始資料的相近程度/差異大小
    '''
    
    pass

    

if __name__ == "__main01__": 
    # print(gamma_pdf_convert(0, 5, 1))
    # 從給定的gamma distribution sample出資料點
    gamma_sample_point = gamma_sampling(57820, 1.9536674, 1/0.6791141, 0)
    draw_sampling_points_hist(gamma_sample_point)
    # 使用sample出的資料點填寫問卷
    # tmp_sample_points = x = np.linspace(0, 50, num=51)
    choice_result_dict_list = []
    for one_questionnaire in [questionnaire_1997, questionnaire_1998, questionnaire_2009]: 
        choice_result_dict = fill_questionnaire(one_questionnaire, gamma_sample_point)
        # print(choice_result_dict)
        choice_result_dict_list.append(choice_result_dict)
    # 問卷選項轉數值
    value_list = [questionnaire_value_1997, questionnaire_value_1998, questionnaire_value_2009]
    new_choice_result_dict_list = []
    for idx in range(len(choice_result_dict_list)): 
        print(choice_result_dict_list[idx])
        new_choice_result_dict_list.append({value_list[idx][k-1]: choice_result_dict_list[idx][k] for k in choice_result_dict_list[idx]})
    print(choice_result_dict_list)
    print(new_choice_result_dict_list)
    for one_questionnaire_result in new_choice_result_dict_list: 
        questionnaire_result_bar(one_questionnaire_result.keys(), one_questionnaire_result.values())
    
    plt.show()

    

if __name__ == "__main__": 
    '''
    快速繪圖 dirty work
    '''
    # 讀取之前整理過的資料，需包含整理好的duration欄位
    with_LTPA_after_screening_df = pd.read_csv('../data/with_LTPA_after_screening.csv')
    duration_1997_df = with_LTPA_after_screening_df.loc[with_LTPA_after_screening_df['yr'] == 1997, 'duration']
    duration_1997_dict = dict(Counter(duration_1997_df))
    print(duration_1997_dict)
    print({k: duration_1997_dict[k] for k in duration_1997_dict if k < 3})
    plt.figure()
    questionnaire_result_bar(duration_1997_dict.keys(), duration_1997_dict.values(), another_fig=False)
    total_sample_points = sum(duration_1997_dict.values())
    total_sample_points_adj = sum({k: duration_1997_dict[k] for k in duration_1997_dict if k < 3}.values())
    print('number of sample points: ', total_sample_points)
    print('number of adj sample points: ', total_sample_points_adj)

    x = np.linspace(0, 50, num=1019)
    print(x)
    alpha = 2.5183706
    beta = 0.4011609
    gamma_p = gamma_pdf_convert(
        x, 
        5, 
        1
    )
    gamma_p2 = gamma_pdf_convert(
        x, 
        alpha, 
        1/beta, 
        0
    )
    # 使用math模組計算，檢驗scipy模組的參數怎麼下才是對的
    # 結論: a = alpha, scale = beta 
    # alpha = 1.380283
    # beta = 1.056349
    test_gamma_p = x**(alpha-1) * math.e**(-1*x/beta) / (math.gamma(alpha) * (beta**alpha))
    
    ax = plt.subplot()
    # ax.plot(x, gamma_p * total_sample_points)
    ax.plot(x, gamma_p2 * total_sample_points, 'm')
    # ax.plot(x, test_gamma_p * total_sample_points_adj, 'y--')
    # ax.plot(x, gamma_p2, 'm')
    # ax.plot(x, test_gamma_p, 'y--')
    plt.show()
    # testing
    print(gamma_pdf_convert(
        questionnaire_value_1997, 
        alpha, 
        1/beta, 
        0
    ) * total_sample_points_adj)


