'''
Configuration of simulation
'''

import pandas as pd 
import numpy as np 
import scipy.stats as st 
from common_module_pkg import gamma_generater   # 這邊可以拿到threshold和midpoint value

class Config_simul: 
    def __init__(self) -> None:
        # Set random seed
        np.random.seed(0)
        self.main_directory = './simul_data/'
        # frozen distribution (scipy distribution given parameters)
        self.distribution_list = [st.gamma(a=1, scale=3), st.gamma(a=1, scale=2.8), st.gamma(a=1, scale=2.7)]  
        self.distribution_nm_list = ['gamma', 'gamma', 'gamma']
        self.threshold_list = [
            [1, 2, 3], 
            [1, 2.5, 4.5, 6.5], 
            [1, 2.5, 4.5, 6.5]
        ]
        self.midpoint_list = [
            [0.5, 1.5, 2.5, 3.0], 
            [0.5, 1.75, 3.5, 5.5, 6.5], 
            [0.5, 1.75, 3.5, 5.5, 6.5]
        ]
        self.num_of_sample_points_list = [57000, 38000, 35000]
        self.version_of_questionnaire_list = [
            [
                lambda a: a<1, 
                lambda a: a>=1 and a<2, 
                lambda a: a>=2 and a<3, 
                lambda a: a>=3
            ], 
            [   
                lambda a: a<1, 
                lambda a: a>=1 and a<2.5, 
                lambda a: a>=2.5 and a<4.5, 
                lambda a: a>=4.5 and a<6.5, 
                lambda a: a>=6.5
            ], 
            [
                lambda a: a<1, 
                lambda a: a>=1 and a<2.5, 
                lambda a: a>=2.5 and a<4.5, 
                lambda a: a>=4.5 and a<6.5, 
                lambda a: a>=6.5
            ]
        ]
        self.cohort_sample_size_list = [20000, 20000]
        self.cohort_habbit_matrix_list = [
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
        self.cohort_first_yr_vec = [
            # Cohort probability vector in 1998 MJ dataset (美兆真實資料的98年(起始年)重複健檢的人口比例)
            np.array([0.455305215, 0.231366991, 0.148282098, 0.079712624, 0.085333073]), 
            # Cohort probability vector in 1999 MJ dataset (美兆真實資料的99年(起始年)重複健檢的人口比例)
            np.array([0.441177793, 0.240278714, 0.156664419, 0.080557429, 0.081321645]) 
        ]

        # ======= ======= ======= ======= ======= ======= =======
        # Do not change these parameters. 
        # These parameter will use in data generation, but won't influence the generation result. 
        # Because probability of bias appearing at each bound is set to 0. 
        # Also the research on these parameters is the future work in the article 
        self.threshold_bias_bound_list_list = [
            [(0.7, 1.3), (1.7, 2.3), (2.3, 3.5)], 
            [(0.7, 1.3), (2.2, 2.8), (4.2, 4.8), (5.5, 7)], 
            [(0.7, 1.3), (2.2, 2.8), (4.2, 4.8), (5.5, 7)]
        ]   # bias發生的範圍 (upper bound, lower bound)
        self.threshold_bias_pdf_list_list = [
            [lambda x: 0, lambda x: 0, lambda x: 0], 
            [lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0], 
            [lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0]
        ]