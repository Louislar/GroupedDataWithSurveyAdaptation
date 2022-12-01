'''
Generate simulation dataset
Cohort is choosen randomly(uniformly) from population's samples
'''

import pandas as pd 
import numpy as np
import scipy.stats as st 
from tqdm import tqdm
from config import Config_simul
from common_module_pkg.gamma_generater import fill_questionnaire

# 產生資料
# 1. 使用多個gamma作為母體隨機產生不同數量的數值 (樣本)
# 2. 讓每個數值填寫固定版本問卷
# 3. 根據threshold和樣本填問卷錯誤的機率，改變問卷填答結果 (Not use in the article)

def multi_origin_samples_generate(multiple_distribution_list, num_of_sample_points_list):
    """
    產生多組母體不同的樣本 (Generate multiple samples from some given distinct distributions)
    Input: 
    :multiple_distribution_list: (list) Distributions that will esed to generate samples 
    :num_of_sample_points_list: (list) Size of sample drawn from each distributions
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
    多組不同母體樣本，填寫問卷 (Fill out the grouped data type questionnaire by sample)
    Input: 
    :sample_df_list: (list) Samples represent the survey responses
    :questionnaire_list: (list) Questionnaires that each sample will answer
    Output: 
    :sample_df_list: (list) DataFrame which includes survey reponse in the "q_result" column
    """

    for i in range(len(sample_df_list)): 
        sample_df_list[i]['q_result'] = fill_questionnaire(
            questionnaire_list[i], 
            sample_df_list[i]['sample'], 
            print_log=False, 
            return_list=True
        )

    return sample_df_list

def set_ID(sample_df_list):
    """
    Give every individual a ID(給每一筆資料一個ID，兩母體間ID也不會重複)
    Input: 
    :sample_df_list: DataFrame with samples 
    Output: 
    :sample_df_list: DataFrame with ID in the 'id' column
    """
    id_count = 0
    for i in range(len(sample_df_list)): 
        sample_df_list[i] = sample_df_list[i].reset_index().rename(columns={'index': 'id'})
        sample_df_list[i].loc[:, 'id'] = sample_df_list[i].loc[:, 'id'] + id_count
        id_count += len(sample_df_list[i]['id'])

    return sample_df_list

def multi_origin_samples_random_change_choice(sample_df_list, bias_bound_list, bias_pdf_list, threshold_list):
    """
    Give some probability that survey responses which is close to threshold change their value. 
    (將填好問卷的多個母體樣本，根據設定好的機率，在threshold附近隨機改動問卷的填答)

    Input: 
    :sample_df_list: DataFrame includes sample and survey responses(in the 'q_result' column)
    :bias_bound_list: Range that will possibly occurs a bias 
    :bias_pdf_list: Probability of a bias occurs
    :threshold_list: A threshold of the grouped data type survey

    Output: 
    :sample_df_list: DataFrames that has been added some bias in the survey response
    """
    def random_switch(datta_val, datta_q_result, pdf, threshold): 
        """
        Use the given pdf to decide whether the response is changing or not
        (單筆資料依照輸入的機率公式、threshold，決定選項要+1還是-1) 

        Input: 
        :datta_val: An real value of a sample(真實連續數值)
        :datta_q_result: The survey response according to the real value(連續數值對應的填答結果)
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

    return sample_df_list

# 將估計的人數取到整數
def make_population_vec_integer(population_vec: np.array, number_of_population: int):
    """
    Round the population vector to decimal
    (將估計出來的人數轉換成整數，但是總數要符合給定的數值，並且要依照小數的大小作為優先加減的依據)

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

def random_pick_cohort_data_and_give_id(sample_df_list, cohort_size, habbit_matrix, second_yr_questionnaire):
    """
    Object: 
        Choose cohort randomly(uniform) from the first year, 
        then use the given transition matrix to compute population vector in second year. 
        (隨機從第一年的人口中挑選做為cohort的人，再根據給定的轉移矩陣，計算出第二年的人口向量，
        根據計算出來的人口向量給予第二年的人口相應的ID
        假設給定的習慣矩陣是要套用再第二年問卷版本的)

    Warning! 
    Remember to check population in second year must greater than cohort in second year!
    Some ID in second year will be modified to ID in first year
    (要檢查第二年的總體人口能不能滿足cohort的人數!!
    注意!! 以第一年的ID為主，修改第二年的ID)
    Input: 
    :sample_df_list: (list)(Only two entry in the list)
    (First DataFrame in the list is first year response, and the Second DF is the second year response) 
    DataFrame with reponses value(in column 'sample') and ID in column 'id'
    :cohort_size: Number of cohort 
    :habbit_matrix: Transition matrix of cohort 
    :second_yr_questionnaire: Questionnaire of the second year
    Output: 
    :sample_df_list: DataFrame includes first year's responses to second year questionnaire 
    in the 'next_yr_ver_q_result' column, also some 'id' in second year will change
    """
    # 0. 讓第一年人口填寫第二年問卷 (為了避免問卷改版的問題) 
    sample_df_list[0]['next_yr_ver_q_result'] = fill_questionnaire(
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
    first_yr_cohort_id_choice_dict = {
        k: first_year_cohort_df.loc[(first_year_cohort_df['next_yr_ver_q_result'] == k), 'id'].values for k in first_yr_cohort_vec_sr.index
    }

    # 2. 利用第一年cohort的人口向量與給定的轉移矩陣，計算出transition matrix內，每一個cell/entry是多少人
    print(habbit_matrix)
    # print(habbit_matrix[:, 0])
    for col_idx in range(habbit_matrix.shape[1]): 
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

    # 3.3 修改第二年資料表當中的ID
    for a_choice in second_yr_cohort_id_choice_dict: 
        choice_criteria = sample_df_list[1]['final_q_result'] == a_choice

        # 只有單個選項的部分 (一份額外的copy)
        a_choice_df = sample_df_list[1].loc[choice_criteria, 'id'].copy()

        # 修正單個選項的ID
        a_choice_df.iloc[:len(second_yr_cohort_id_choice_dict[a_choice])] = second_yr_cohort_id_choice_dict[a_choice]

        # 修正好的單選項資料表，貼回原始的大資料表
        sample_df_list[1].loc[choice_criteria, 'id'] = a_choice_df

    return sample_df_list

# Original simulation
if __name__ == "__main01__":
    simulationConfig = Config_simul()
    sample_point_df_list = multi_origin_samples_generate(simulationConfig.distribution_list, simulationConfig.num_of_sample_points_list)
    sample_point_df_list = multi_origin_samples_fill_questionnaire(sample_point_df_list, simulationConfig.version_of_questionnaire_list)
    sample_point_df_list = multi_origin_samples_random_change_choice(sample_point_df_list, simulationConfig.threshold_bias_bound_list_list, simulationConfig.threshold_bias_pdf_list_list, simulationConfig.threshold_list)
    
    # 設定ID (從0開始一直+1上去)
    sample_point_df_list = set_ID(sample_point_df_list)

    # 隨機挑選cohort人口，並且給定ID
    sample_point_df_list[0:2] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[0:2], 
            simulationConfig.cohort_sample_size_list[0], 
            simulationConfig.cohort_habbit_matrix_list[0], 
            simulationConfig.version_of_questionnaire_list[1]
        )
    sample_point_df_list[1:3] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[1:3], 
            simulationConfig.cohort_sample_size_list[1], 
            simulationConfig.cohort_habbit_matrix_list[1], 
            simulationConfig.version_of_questionnaire_list[2]
        )

    for i in range(len(sample_point_df_list)): 
        sample_point_df_list[i].to_csv(simulationConfig.main_directory+'year_{0}.csv'.format(i), index=False)
        

# Simulation with random generate transition matrices
# Output directory simul_data_random_{index of the random transition matrix}/
if __name__ == "__main__":
    simulationConfig = Config_simul()

    ## random transition matrix index {1, 2, 3}
    rndTransMatInd = 3
    rndTransMatCount = 3
    ## Change output path
    simulationConfig.main_directory='./simul_data_{0}/'.format(rndTransMatInd)
    ## read random generate transition matrices
    rndGenTransMat = pd.read_csv(
        'randomTransitionMatrices/random_transition_matrix_{0}.csv'.format(rndTransMatInd-1)
    ).values
    rndGenTransMatNext = pd.read_csv(
        'randomTransitionMatrices/random_transition_matrix_{0}.csv'.format(rndTransMatInd%rndTransMatCount)
    ).values
    print('The random gen trans mat: ')
    print(rndGenTransMat)
    
    ## Change transition matrix used in simulation to those randomly generated
    ## Change the second transition matrix to another random generate matrix for convenience 
    simulationConfig.cohort_habbit_matrix_list[0] = rndGenTransMat
    simulationConfig.cohort_habbit_matrix_list[1] = rndGenTransMatNext
    print(simulationConfig.cohort_habbit_matrix_list)
    

    sample_point_df_list = multi_origin_samples_generate(simulationConfig.distribution_list, simulationConfig.num_of_sample_points_list)
    sample_point_df_list = multi_origin_samples_fill_questionnaire(sample_point_df_list, simulationConfig.version_of_questionnaire_list)
    sample_point_df_list = multi_origin_samples_random_change_choice(sample_point_df_list, simulationConfig.threshold_bias_bound_list_list, simulationConfig.threshold_bias_pdf_list_list, simulationConfig.threshold_list)
    
    # 設定ID (從0開始一直+1上去)
    sample_point_df_list = set_ID(sample_point_df_list)

    # 隨機挑選cohort人口，並且給定ID
    sample_point_df_list[0:2] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[0:2], 
            simulationConfig.cohort_sample_size_list[0], 
            simulationConfig.cohort_habbit_matrix_list[0], 
            simulationConfig.version_of_questionnaire_list[1]
        )
    sample_point_df_list[1:3] = \
        random_pick_cohort_data_and_give_id(
            sample_point_df_list[1:3], 
            simulationConfig.cohort_sample_size_list[1], 
            simulationConfig.cohort_habbit_matrix_list[1], 
            simulationConfig.version_of_questionnaire_list[2]
        )

    for i in range(len(sample_point_df_list)): 
        sample_point_df_list[i].to_csv(simulationConfig.main_directory+'year_{0}.csv'.format(i), index=False)