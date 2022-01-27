'''
增加隨機性的simulation data產生器
'''

import pandas as pd 
import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
from collections import Counter 
from common_module_pkg import gamma_generater   # 這邊可以拿到threshold和midpoint value
from tqdm import tqdm

main_directory = './simul_data/'

# 給定參數
# pre 1. 給定random state
# 1. 每一年gamma的參數
# 2. 每一年的錯誤機率分布 (每一個threshold的錯誤機率分布 或是某個固定機率轉移 以及錯誤發生範圍[upper bound, lower bound] )
# 3. 每一年的人數/每一年的樣本數
# 4. 每一年的問卷版本
# 5. 兩年間的重複抽樣人口
# 6. 兩年間的習慣矩陣

# ref: https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed/49557099
# ref: https://stackoverflow.com/questions/16016959/scipy-stats-seed 
np.random.seed(0)

distribution_list = [st.gamma(a=1, scale=3), st.gamma(a=1, scale=2.8), st.gamma(a=1, scale=2.7)]  # frozen的distribution (給好參數的scipy distribution)
# distribution_list = [st.lognorm(s=1, scale=3.0326532, loc=0), st.lognorm(s=1, scale=2.4261226, loc=0), st.lognorm(s=1, scale=1.819591977, loc=0)]  # frozen的distribution (給好參數的scipy distribution)
# distribution_list = [st.expon(scale=3), st.expon(scale=2.8), st.expon(scale=2.7)]
# distribution_list = [st.weibull_min(c=1, scale=3), st.weibull_min(c=1, scale=2.8), st.weibull_min(c=1, scale=2.7)]
# distribution_list = [st.lognorm(s=1, scale=1.819591979, loc=0), st.lognorm(s=1, scale=1.69828584719, loc=0), st.lognorm(s=1, scale=1.637632781224, loc=0)]
threshold_bias_bound_list_list = [
    [(0.7, 1.3), (1.7, 2.3), (2.3, 3.5)], 
    [(0.7, 1.3), (2.2, 2.8), (4.2, 4.8), (5.5, 7)], 
    [(0.7, 1.3), (2.2, 2.8), (4.2, 4.8), (5.5, 7)]
]   # bias發生的範圍 (upper bound, lower bound)
# threshold_bias_pdf_list_list = [
#     [lambda x: 0.05, lambda x: 0.05, lambda x: 0.1], 
#     [lambda x: 0.05, lambda x: 0.05, lambda x: 0.05, lambda x: 0.1], 
#     [lambda x: 0.05, lambda x: 0.05, lambda x: 0.05, lambda x: 0.1]
# ]         # bias發生的機率分布 (e.g. uniform, normal) (給frozen p.d.f 或是給一個函數回傳固定機率) (以threshold=0為中心，往外延伸的)
threshold_bias_pdf_list_list = [
    [lambda x: 0, lambda x: 0, lambda x: 0], 
    [lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0], 
    [lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0]
]  
threshold_list = [
    [1, 2, 3], 
    [1, 2.5, 4.5, 6.5], 
    [1, 2.5, 4.5, 6.5]
]
num_of_sample_points_list = [57000, 38000, 35000]
version_of_questionnaire_list = [gamma_generater.questionnaire_1997, gamma_generater.questionnaire_1998, gamma_generater.questionnaire_1998]
cohort_sample_size_list = [20000, 20000]
cohort_habbit_matrix_list = [
    # 1998 to 1999 habbit matrix
    np.array([
        [0.68580936, 0.288550908, 0.142056691, 0.095033722, 0.068155785], 
        [0.200193216, 0.417405999, 0.264337508, 0.126916002, 0.078465063], 
        [0.070094461, 0.194127588, 0.377059987, 0.285714286, 0.112829324], 
        [0.024581365, 0.063160118, 0.142386289, 0.293071735, 0.220504009], 
        [0.019321597, 0.036755387, 0.074159525, 0.199264255, 0.520045819]
    ]), 
    # 1999 to 2000 habbit matrix
    np.array([
        [0.686977787, 0.28868101, 0.122238164, 0.079799107, 0.071310116], 
        [0.209394742, 0.421702526, 0.28723099, 0.138392857, 0.091763405], 
        [0.066231914, 0.205612722, 0.371879484, 0.267857143, 0.142067441], 
        [0.021805584, 0.05313377, 0.147776184, 0.310267857, 0.188501935], 
        [0.015589974, 0.030869972, 0.070875179, 0.203683036, 0.506357103]
    ])
]
cohort_first_yr_vec = [
    np.array([0.455305215, 0.231366991, 0.148282098, 0.079712624, 0.085333073]),      # 每兆真實資料的98年(起始年)重複健檢的人口比例
    np.array([0.441177793, 0.240278714, 0.156664419, 0.080557429, 0.081321645])       # 每兆真實資料的99年(起始年)重複健檢的人口比例
]

# 產生資料
# 1. 使用多個gamma作為母體隨機產生不同數量的數值 (樣本)
# 2. 讓每個數值填寫固定版本問卷
# 3. 根據threshold和樣本填問卷錯誤的機率，改變問卷填答結果

def multi_origin_samples_generate(multiple_distribution_list, num_of_sample_points_list):
    """
    產生多組母體不同的樣本
    Output: 
    :sample_df_list: (list) 每一個母體的樣本，使用dataframe儲存
    """
    sample_df_list = []
    for i in range(len(multiple_distribution_list)): 
        sample_points_arr = multiple_distribution_list[i].rvs(size=num_of_sample_points_list[i])
        sample_df_list.append(pd.DataFrame({'sample': sample_points_arr}))

    return sample_df_list

def multi_origin_samples_fill_questionnaire(sample_df_list, questionnaire_list):
    """
    多組不同母體樣本，填寫問卷
    """

    for i in range(len(sample_df_list)): 
        sample_df_list[i]['q_result'] = gamma_generater.fill_questionnaire(
            questionnaire_list[i], 
            sample_df_list[i]['sample'], 
            print_log=False, 
            return_list=True
        )

    return sample_df_list
    
def multi_origin_samples_random_change_choice(sample_df_list, bias_bound_list, bias_pdf_list, threshold_list):
    """
    將填好問卷的多個母體樣本，根據設定好的機率，在threshold附近隨機改動問卷的填答
    """
    def random_switch(datta_val, datta_q_result, pdf, threshold): 
        """
        單筆資料依照輸入的機率公式、threshold，決定選項要+1還是-1 

        Input: 
        :datta_val: 真實連續數值
        :datta_q_result: 連續數值對應的填答結果
        """
        # print(datta_val)
        # print(threshold)
        change_probability = pdf(datta_val) 
        is_change = st.bernoulli(change_probability).rvs(size=1)[0] == 1 
        if is_change: 
            if datta_val-threshold < 0: 
                datta_q_result += 1
            elif datta_val-threshold > 0:
                datta_q_result -= 1
        return datta_q_result 

    for i in tqdm(range(len(sample_df_list))): 
        # 每一個threshold有ub, lb，且都會做一次random switch 
        for j in tqdm(range(len(bias_bound_list[i]))): 
            in_bound_criterion = (sample_df_list[i]['sample'] >= bias_bound_list[i][j][0]) & (sample_df_list[i]['sample'] <= bias_bound_list[i][j][1])
            sample_df_list[i].loc[in_bound_criterion,'new_q_result'] = \
                sample_df_list[i].loc[in_bound_criterion, :].apply(lambda x: random_switch(
                    x['sample'], 
                    x['q_result'], 
                    bias_pdf_list[i][j], 
                    threshold_list[i][j]
                    ), axis=1)
            # print('number of potential change sample points: ', in_bound_criterion.sum())
            # break
            pass
        # break
    
        # 合併
        sample_df_list[i].loc[sample_df_list[i]['new_q_result'].isna(), 'final_q_result'] = \
            sample_df_list[i].loc[sample_df_list[i]['new_q_result'].isna(), 'q_result']
        sample_df_list[i].loc[sample_df_list[i]['new_q_result'].notna(),'final_q_result'] = \
           sample_df_list[i].loc [sample_df_list[i]['new_q_result'].notna(), 'new_q_result']
    pass

    return sample_df_list

def set_ID(sample_df_list):
    """
    給每一筆資料一個ID，兩母體間ID也不會重複
    """
    id_count = 0
    for i in range(len(sample_df_list)): 
        sample_df_list[i] = sample_df_list[i].reset_index().rename(columns={'index': 'id'})
        sample_df_list[i].loc[:, 'id'] = sample_df_list[i].loc[:, 'id'] + id_count
        id_count += len(sample_df_list[i]['id'])

    return sample_df_list

def use_habbit_matrix_to_gen_id(sample_df_list, cohort_size, habbit_matrix, second_yr_questionnaire, yr_population_vec_list):
    """
    Object: give cohort data corresponding IDs
    目前只處理"兩個"年份的cohort ID對齊，並且不會更動到第一個年份的ID，會更動的是第二個年份的ID
    需要指定部分人有重複填寫問卷，並且要依照給定的習慣矩陣的比例，分配這些重複填寫問卷的人
    (目前作法) 假設第一年的問卷與第二年的是不同版本，並且第二年與第三年是同一版本的，
                所以第一年還需要多填寫第二年的問卷版本才行，column名稱定為'next_yr_ver_q_result' 
                這種方法對於沒有問卷改版的情況也適用，只要在參數輸入時給同樣版本的問卷即可

    Input: 
    :sample_df_list: (list) (pd.DataFrame) 有原始數值、問卷填答結果、加雜訊的問卷填答結果
    :cohort_size: 兩年間的重複健檢人口數有多少
    :habbit_matrix: (np.array) 兩年間重複健檢人口的習慣矩陣 
    :second_yr_questionnaire: 第二個年份的問卷版本 (怕有問卷改版的情況)
    :first_yr_population_vec: 第一個年份的人數比例向量

    Output: 
    :sample_df_list: 修正好cohort ID欄位的資料表(DataFrame)
    """
    # 第一年先填第二年(98年)問卷 (新column name: 'next_yr_ver_q_result')
    sample_df_list[0]['next_yr_ver_q_result'] = gamma_generater.fill_questionnaire(
            second_yr_questionnaire, 
            sample_df_list[0]['sample'], 
            print_log=False, 
            return_list=True
        )
        
    # 先把第一年(97年)的20000個cohort人挑出來，並且計算人數比例 (非隨機挑選，要依照輸入的人數比例挑選)
    # 各選項應該要挑選的人數
    first_year_cohort_population_count = np.round(yr_population_vec_list * cohort_size)
    # print(first_year_cohort_population_count)
    # 小數修正回整數，少的補回最後一個選項
    population_loss = cohort_size - np.sum(first_year_cohort_population_count) 
    first_year_cohort_population_count[-1] += population_loss
    print('理論上第一年各個選項要有多少cohort的人: ', first_year_cohort_population_count)
    # print(population_loss)

    first_yr_population_count = Counter(sample_df_list[0]['next_yr_ver_q_result'])
    first_yr_population_count_dict = {k: first_yr_population_count[k] for k in sorted(first_yr_population_count)}
    first_yr_population_count_list = list(first_yr_population_count_dict.values())
    print('第一年各個選項實際人口數: ', first_yr_population_count_list)

    # 檢查cohort資料的各個選項人數有沒有超過總體各選項人數 
    for idx in range(len(first_year_cohort_population_count)): 
        if first_year_cohort_population_count[idx] > first_yr_population_count_list[idx]: 
            print('(first year) population less than cohort data.')
            print('population count list: ', first_yr_population_count_list)
            print('cohort count list: ', first_year_cohort_population_count)
            return 0
    

    
    # 把習慣矩陣從比例的形式，轉換成實際人數 (會用到第一年(97年)的人口vector)
    for i in range(habbit_matrix.shape[1]): 
        habbit_matrix[:, i] = np.round(habbit_matrix[:, i] * first_year_cohort_population_count[i])
        # 因為數值相乘後會有小數，所以直接捨去小數點後的數值，最後加總不夠原本的總人數數的話，要補齊人數，總人數要維持一樣
        # 單行補值，如果人數不夠的話
        loss_population = first_year_cohort_population_count[i] - sum(habbit_matrix[:, i])
        habbit_matrix[-1, i] += loss_population
        # print(sum(habbit_matrix_list:, i]))
        # print(first_year_cohort_population_count[i])


    # 檢查看看98年的各選項人數夠不夠
    second_yr_q_result_count_theo = [np.sum(habbit_matrix[i, :]) for i in range(habbit_matrix.shape[0])] # 理論上每一個選項的數量
    print('理論上第二年各個選項要有多少cohort的人: ', second_yr_q_result_count_theo)

    print(sample_df_list[1])
    second_yr_population_q_count = Counter(sample_df_list[1]['final_q_result'])
    second_yr_population_q_count_dict = {k: second_yr_population_q_count[k] for k in sorted(second_yr_population_q_count)}
    second_yr_population_q_count_list = list(second_yr_population_q_count_dict.values())
    print(second_yr_population_q_count)
    print('第二年各個選項實際人口數: ', second_yr_population_q_count_list)

    for idx in range(len(second_yr_q_result_count_theo)): 
        if second_yr_q_result_count_theo[idx] > second_yr_population_q_count_list[idx]: 
            print('(second year) population less than cohort data.')
            print('population count: ', second_yr_population_q_count_list)
            print('cohort count: ', second_yr_q_result_count_theo)
            return 0
        
    
    # 以上就確認完cohort的資料不會比population的資料還要多，就可以確保cohort的ID可以完整被assign
    # =========================================================

    # 給每一筆資料都assign ID (考慮在這個function之前就做好) (用set_ID做完了)
    # 從第一年當中挑出要做為cohort的資料，把這些ID記錄下來
    first_yr_id_dict = {}
    for idx in range(len(first_year_cohort_population_count)): 
        choice_criteria = sample_df_list[0]['next_yr_ver_q_result'] == (idx+1) # 選某個選項的條件式
        choice_id_arr = sample_df_list[0].loc[choice_criteria, 'id'].values
        # print(choice_id_sr.shape)
        # print(first_year_cohort_population_count[idx])
        first_yr_id_dict[idx+1] = choice_id_arr[range(int(first_year_cohort_population_count[idx]))]

    # 根據transition matrix上的各個cell的人數分配ID
    # 從第一年的id列表中分派到第二年的某個選項列表
    # print(habbit_matrix)
    assign_to_second_yr_choice_id_dict = {(i+1): [] for i in range(habbit_matrix.shape[1])} # key: 第二年的選項, values: 分配給key的ID
    for first_yr_idx in range(habbit_matrix.shape[0]): 
        first_yr_assigned_id_count = 0
        for second_yr_idx in range(habbit_matrix.shape[1]): 
            
            # print('第一個年份的id: ', first_yr_id_dict[first_yr_idx+1][first_yr_assigned_id_count: int(first_yr_assigned_id_count + habbit_matrix_list[second_yr_idx, first_yr_idx])])
            if second_yr_idx+1 == 5: 
                # print('應該要分配到第二年的ID人數: ', habbit_matrix_list[second_yr_idx, first_yr_idx])
                # print(len(first_yr_id_dict[first_yr_idx+1][first_yr_assigned_id_count: int(first_yr_assigned_id_count + habbit_matrix_list[second_yr_idx, first_yr_idx])]))
                # print(int(first_yr_assigned_id_count + habbit_matrix_list[second_yr_idx, first_yr_idx]))
                # print(len(first_yr_id_dict[second_yr_idx+1]))
                pass 
            assign_to_second_yr_choice_id_dict[second_yr_idx+1].extend(
                first_yr_id_dict[first_yr_idx+1][first_yr_assigned_id_count: int(first_yr_assigned_id_count + habbit_matrix[second_yr_idx, first_yr_idx])]
            ) 
            
            first_yr_assigned_id_count += int(habbit_matrix[second_yr_idx, first_yr_idx])
        pass
    
    # print({k: len(assign_to_second_yr_choice_id_dict[k]) for k in assign_to_second_yr_choice_id_dict})

    # 依照求出的ID列表，更改第二年的ID欄位
    second_yr_group = sample_df_list[1].groupby(['final_q_result'])
    for a_2_yr_group in second_yr_group: 
        group_criteria = (sample_df_list[1]['final_q_result'] == a_2_yr_group[0])
        print('\n哪一個選項的人: ', a_2_yr_group[0])
        print('那一個選項是cohort有多少: ', len(assign_to_second_yr_choice_id_dict[a_2_yr_group[0]]))
        print('cohort的ID是多少(前10個): ', assign_to_second_yr_choice_id_dict[a_2_yr_group[0]][:10])
        
        # 修正cohort的ID
        index_criteria = sample_df_list[1].index.isin(
            sample_df_list[1][group_criteria].index[:len(assign_to_second_yr_choice_id_dict[a_2_yr_group[0]])]
        )
        sample_df_list[1].loc[group_criteria & index_criteria, 'id'] = \
            assign_to_second_yr_choice_id_dict[a_2_yr_group[0]]

    return sample_df_list


def random_pick_cohort_data_and_give_id(sample_df_list, cohort_size, habbit_matrix, second_yr_questionnaire):
    """
    Object: 
        隨機從第一年的人口中挑選做為cohort的人，再根據給定的轉移矩陣，計算出第二年的人口向量，
        根據計算出來的人口向量給予第二年的人口相應的ID
        假設給定的習慣矩陣是要套用再第二年問卷版本的

    要檢查第二年的總體人口能不能滿足cohort的人數!!
    注意!! 以第一年的ID為主，修改第二年的ID
    """
    # 0. 讓第一年人口填寫第二年問卷 (為了避免問卷改版的問題) 
    sample_df_list[0]['next_yr_ver_q_result'] = gamma_generater.fill_questionnaire(
            second_yr_questionnaire, 
            sample_df_list[0]['sample'], 
            print_log=False, 
            return_list=True
        )

    # 1. 從第一年隨機挑選cohort的人
    first_year_cohort_df = sample_df_list[0].sample(n=cohort_size)
    # 1.1 計算第一年cohort的人口向量
    first_yr_cohort_vec_sr = first_year_cohort_df['next_yr_ver_q_result'].value_counts().sort_index()
    first_yr_cohort_vec_arr = first_yr_cohort_vec_sr.values
    print('第一年cohort人口: ', first_yr_cohort_vec_arr)
    print('第一年pop人口: ', sample_df_list[0]['next_yr_ver_q_result'].value_counts().sort_index().values)
    # 1.2 分類第一年人口id，每個ID都屬於某個選項
    # first_yr_cohort_choice_dict = {k: first_yr_cohort_vec_sr.loc[k] for k in first_yr_cohort_vec_sr.index}
    first_yr_cohort_id_choice_dict = {
        k: first_year_cohort_df.loc[(first_year_cohort_df['next_yr_ver_q_result'] == k), 'id'].values for k in first_yr_cohort_vec_sr.index
    }

    # for k in first_yr_cohort_id_choice_dict: 
    #     print(first_yr_cohort_id_choice_dict[k])
    # print(first_yr_cohort_choice_dict)




    # 2. 利用第一年cohort的人口向量與給定的轉移矩陣，計算出transition matrix內，每一個cell/entry是多少人
    print(habbit_matrix)
    # print(habbit_matrix[:, 0])
    for col_idx in range(habbit_matrix.shape[1]): 
        # print(habbit_matrix[:, col_idx])
        # print(habbit_matrix[:, col_idx] * first_yr_cohort_vec_arr[col_idx])
        # print(
        #     make_population_vec_integer(habbit_matrix[:, col_idx] * first_yr_cohort_vec_arr[col_idx], first_yr_cohort_vec_arr[col_idx])
        # )
        habbit_matrix[:, col_idx] = make_population_vec_integer(habbit_matrix[:, col_idx] * first_yr_cohort_vec_arr[col_idx], first_yr_cohort_vec_arr[col_idx])

    habbit_matrix = habbit_matrix.astype(int)
    print(habbit_matrix)

    # 3. 根據計算出的每一個cell/entry是多少人，給予第二年的cohort人口ID
    # 3.1 檢查第二年全體人口的人數有沒有大於cohort的人數
    second_yr_pop_vec_sr = sample_df_list[1]['final_q_result'].value_counts().sort_index()
    second_yr_pop_vec_arr = second_yr_pop_vec_sr.values
    second_yr_cohort_vec = np.sum(habbit_matrix, axis=1)
    print('第二年cohort人口: ', second_yr_cohort_vec)
    print('第二年pop人口: ', second_yr_pop_vec_arr)

    for idx in range(len(second_yr_cohort_vec)): 
        if second_yr_cohort_vec[idx] > second_yr_pop_vec_arr[idx]: 
            print('cohort number larger than population.')
            print('population vec: ', second_yr_pop_vec_arr)
            print('cohort vec: ', second_yr_cohort_vec)

    # 3.2 分配ID給第二年的各個選項
    second_yr_cohort_id_choice_dict = {(i+1): [] for i in range(habbit_matrix.shape[1])}
    for first_yr_choice_idx in range(habbit_matrix.shape[1]): 
        tmp_index_count = 0
        for second_yr_choice_idx in range(habbit_matrix.shape[0]): 
            second_yr_cohort_id_choice_dict[second_yr_choice_idx+1].extend(
                first_yr_cohort_id_choice_dict[first_yr_choice_idx+1][tmp_index_count:tmp_index_count+habbit_matrix[second_yr_choice_idx, first_yr_choice_idx]]
            )
            tmp_index_count += habbit_matrix[second_yr_choice_idx, first_yr_choice_idx]

    # print(second_yr_cohort_id_choice_dict)
    # for k in second_yr_cohort_id_choice_dict: 
    #     print(len(second_yr_cohort_id_choice_dict[k]))
    # print(np.sum(habbit_matrix, axis=1))

    # 3.3 修改第二年資料表當中的ID
    for a_choice in second_yr_cohort_id_choice_dict: 
        choice_criteria = sample_df_list[1]['final_q_result'] == a_choice

        # 只有單個選項的部分 (一份額外的copy)
        a_choice_df = sample_df_list[1].loc[choice_criteria, 'id'].copy()

        # 修正單個選項的ID
        a_choice_df.iloc[:len(second_yr_cohort_id_choice_dict[a_choice])] = second_yr_cohort_id_choice_dict[a_choice]

        # print(sample_df_list[1].head(20))
        # 修正好的單選項資料表，貼回原始的大資料表
        sample_df_list[1].loc[choice_criteria, 'id'] = a_choice_df

        # print(sample_df_list[1].head(20))

    return sample_df_list

# 將估計的人數取到整數
def make_population_vec_integer(population_vec: np.array, number_of_population: int):
    """
    將估計出來的人數轉換成整數，但是總數要符合給定的數值，並且要依照小數的大小作為優先加減的依據

    Input: 
    :population_vec: 估計出來帶有小數的人數比例向量
    :number_of_population: 總體人數數量
    """
    
    round_arr = np.floor(population_vec).astype(int)

    sum_of_round_arr = np.sum(round_arr).astype(int)

    number_need_to_add = number_of_population - sum_of_round_arr

    decimal_arr = population_vec - round_arr

    sorted_decimal_array_idx = np.argsort(decimal_arr)[::-1]

    add_1_idx = sorted_decimal_array_idx[:number_need_to_add]
 
    round_arr[add_1_idx] += 1

    return round_arr 

# ======= ======= ======= ======= ======= ======= =======
# 檢查simulation的結果有沒有正確
def cohort_data_retrieve(datta_df_list, is_ver_of_q_change=False): 
    """
    拿到重複健檢的人的資料
    """
    first_yr_data = datta_df_list[0]
    second_yr_data = datta_df_list[1]

    first_yr_cohort_criteria = first_yr_data['id'].isin(second_yr_data['id'])
    second_yr_cohort_criteria = second_yr_data['id'].isin(first_yr_data['id'])

    first_yr_cohort = first_yr_data[first_yr_cohort_criteria]
    second_yr_cohort = second_yr_data[second_yr_cohort_criteria]
    
    print('cohort數量 1: ', first_yr_cohort_criteria.sum())
    print('cohort數量 2: ', second_yr_cohort_criteria.sum())

    # 檢查轉移矩陣
    merged_df = pd.merge(first_yr_cohort, second_yr_cohort, how='inner', on='id')
    if is_ver_of_q_change == False: 
        merged_df = merged_df.rename(columns={'final_q_result_x': 'first_yr_choice', 'final_q_result_y': 'second_yr_choice'})
        merged_df = merged_df[['first_yr_choice', 'second_yr_choice']]
    if is_ver_of_q_change == True: 
        merged_df = merged_df.rename(columns={'final_q_result_x': 'first_yr_choice', 'final_q_result_y': 'second_yr_choice', 'next_yr_ver_q_result_x': 'first_yr_value_next_yr_q'})
        merged_df = merged_df[['first_yr_choice', 'second_yr_choice', 'first_yr_value_next_yr_q']]
    # print(merged_df.head(10))
    # print(merged_df.shape)
    
    # 人數版的轉移矩陣
    transition_matrix = pd.crosstab(merged_df['second_yr_choice'], merged_df['first_yr_choice'], margins=True)
    if is_ver_of_q_change == True: 
        transition_matrix = pd.crosstab(merged_df['second_yr_choice'], merged_df['first_yr_value_next_yr_q'], margins=True)
    print(transition_matrix)

    # 比例版的轉移矩陣
    transition_matrix = pd.crosstab(merged_df['second_yr_choice'], merged_df['first_yr_choice'], margins=True, normalize='columns')
    if is_ver_of_q_change == True: 
        transition_matrix = pd.crosstab(merged_df['second_yr_choice'], merged_df['first_yr_value_next_yr_q'], margins=True, normalize='columns')
    print(transition_matrix)

    # 檢查cohort第一年人數比例
    first_yr_q_result_dict = dict(Counter(first_yr_cohort['final_q_result']))
    if is_ver_of_q_change == True: 
        first_yr_q_result_dict = dict(Counter(first_yr_cohort['next_yr_ver_q_result']))
    first_yr_q_result_dict = {k: first_yr_q_result_dict[k] for k in sorted(first_yr_q_result_dict)}
    first_yr_q_result_arr = np.array(list(first_yr_q_result_dict.values()))
    print('第一年的cohort人數比例: ', first_yr_q_result_arr / first_yr_q_result_arr.sum())

    # 檢查全體第一年人數比例
    first_yr_q_result_dict = dict(Counter(first_yr_data['final_q_result']))
    if is_ver_of_q_change == True: 
        first_yr_q_result_dict = dict(Counter(first_yr_data['next_yr_ver_q_result']))
    first_yr_q_result_dict = {k: first_yr_q_result_dict[k] for k in sorted(first_yr_q_result_dict)}
    first_yr_q_result_arr = np.array(list(first_yr_q_result_dict.values()))
    print('第一年的全體人數比例: ', first_yr_q_result_arr / first_yr_q_result_arr.sum())

    pass

# ======= ======= ======= ======= ======= ======= =======
# 不指定cohort人口比例
if __name__ == "__main__":
    sample_point_df_list = multi_origin_samples_generate(distribution_list, num_of_sample_points_list)
    sample_point_df_list = multi_origin_samples_fill_questionnaire(sample_point_df_list, version_of_questionnaire_list)
    sample_point_df_list = multi_origin_samples_random_change_choice(sample_point_df_list, threshold_bias_bound_list_list, threshold_bias_pdf_list_list, threshold_list)
    
    # 設定ID (從0開始一直+1上去)
    sample_point_df_list = set_ID(sample_point_df_list)

    # 隨機挑選cohort人口，並且給定ID
    sample_point_df_list[0:2] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[0:2], 
            cohort_sample_size_list[0], 
            cohort_habbit_matrix_list[0], 
            version_of_questionnaire_list[1]
        )
    sample_point_df_list[1:3] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[1:3], 
            cohort_sample_size_list[1], 
            cohort_habbit_matrix_list[1], 
            version_of_questionnaire_list[2]
        )

    for i in range(len(sample_point_df_list)): 
        sample_point_df_list[i].to_csv(main_directory+'year_{0}.csv'.format(i), index=False)
        pass
    
# ======= ======= ======= ======= ======= ======= =======
# 指定cohort 人口比例的方式
if __name__ == "__main01__": 
    sample_point_df_list = multi_origin_samples_generate(distribution_list, num_of_sample_points_list)
    sample_point_df_list = multi_origin_samples_fill_questionnaire(sample_point_df_list, version_of_questionnaire_list)
    sample_point_df_list = multi_origin_samples_random_change_choice(sample_point_df_list, threshold_bias_bound_list_list, threshold_bias_pdf_list_list, threshold_list)
    
    # 設定ID (從0開始一直+1上去)
    sample_point_df_list = set_ID(sample_point_df_list)
    # for i in sample_point_df_list: 
    #     print(i.iloc[0, :])
    #     print(i.iloc[-1, :])

    # 設定cohort ID
    # 設定第一和第二個母體人口的cohort ID
    sample_point_df_list[0:2] = use_habbit_matrix_to_gen_id(sample_point_df_list[0:2], cohort_sample_size_list[0], cohort_habbit_matrix_list[0], version_of_questionnaire_list[1], cohort_first_yr_vec[0])
    # 設定第二和第三個母體人口的cohort ID
    sample_point_df_list[1:3] = use_habbit_matrix_to_gen_id(sample_point_df_list[1:3], cohort_sample_size_list[1], cohort_habbit_matrix_list[1], version_of_questionnaire_list[2], cohort_first_yr_vec[1])

    # print(sample_point_df_list[0].head(10))
    # print(sample_point_df_list[0].shape)
    # print(((sample_point_df_list[0]['new_q_result'].notna()) & (sample_point_df_list[0]['q_result'] != sample_point_df_list[0]['new_q_result'])).sum())

    # output成多個csv檔案
    for i in range(len(sample_point_df_list)): 
        sample_point_df_list[i].to_csv(main_directory+'year_{0}.csv'.format(i), index=False)
        pass


# ======= ======= ======= ======= ======= ======= ======= 
# 檢查simulation的結果有沒有正確 
if __name__ == "__main01__": 
    sample_df_list = []
    for i in range(len(distribution_list)): 
        sample_df_list.append(
            pd.read_csv('./simulation_with_random_switch/year_{0}.csv'.format(i))
        )

    # 檢查cohort數量、matrix
    cohort_data_retrieve(sample_df_list[0:2], True)
    cohort_data_retrieve(sample_df_list[1:3])