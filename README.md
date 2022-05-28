# GroupedDataWithSurveyAdaptation

[![hackmd-github-sync-badge](https://hackmd.io/deR1Z80sTfee2bE8WDjZ5Q/badge)](https://hackmd.io/deR1Z80sTfee2bE8WDjZ5Q)

## Index
[TOC]

## Overview

This program will estimate the transition matrix according to revision by cohort data, then apply it to samples of population for aligning versions of questionnaire. 

The detail introduction to this program can refer to this article ...

## Environment

Windows 10

### Python

- Scipy
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- rpy2

### R

Reminder: R is installed with rpy2(python package list above), but the package list below need to be installed manually
- Nloptr

### Matlab

- R2020a
- Optimization toolbox version 8.5

## Usage
<!-- TODO: 整理成一個bat檔案? -->
Please follow the execution order below to reproduce the result that we report in the article. 

Please note that the R code is integrate in the python code, so there is no need to execute R code in a separate R console. On the contrary, the Matlab code(main_qp.m) needs to be executed in a separate Matlab console, since it's code and interpreter are not integrated in the python. 

1. Check parameters in the config.py. 

<!-- TODO: 執行完後會給你甚麼? e.g. simulation dataset -->
2. Execute generate_simul_data.py

<!-- TODO: 加入理想改版矩陣的格式(.csv內的格式) -->
3. Hand craft the ideal revision matrix name it `Python_G_matrix.csv` and put it in `/qp_input_output/`. The format of `Python_G_matrix.csv` please refer to the pre-generated dataset. 

4. Execute main_qp.m in a Matlab console

5. Execute estimation.py

## Simulation code and dataset

### Code

1. config.py
    - Set simulation parameters here. These parameter will be used by all the other python scripts. 

2. generate_simul_data.py 
    - Generate simulation dataset. There is a config.py that can set parameters of the simulation.
    - Output
        - Multi-year simulation data: `/[main_directory in config.py]/year_[number_of_year].csv`

3. preprocessing_for_qp.py
    - Compute matrices and vectors for Quadratice Programming. 
    - Output
        <!-- TODO: output列表整理在額外的.csv會比較好 -->
        1. Under `/[main_directory in config.py]/matrix_and_vector/`
            - List in the preprocess_for_qp_output.csv(In the Root folder)

        2. Under `/[main_directory in config.py]/qp_input_output/`
            - Cohort p.v. in the first year: `python_vector_c.csv`
            - Cohort p.v. in the second year: `python_f_vec.csv`
            - Transition matrix of cohort at first year: `python_T_matrix.csv`
            - Time-related matrix of cohort at first year: `python_M_matrix.csv`

4. main_qp.m
    - Solve the Quadratice Programming
    - Output(All under `/[main_directory in config.py]/qp_input_output/`): 
        - The four parameter of similarity: `matlab_four_param.csv` 
        - The estimate time-related matrix of cohort at first year: `matlab_habbit_matrix.csv`
        - The estimate revision-related matrix at first year: `matlab_version_change_matrix.csv`
        - The estimate first cohort at first year response to the questionnaire used in the second year :`matlab_new_cohort_97_vec`

5. estimation.py
    - Compute the mean estimation by Gamma fit, midpoint method, QP(with midpoint method) and QP(with Gamma fit)
    - Compute the estimated revisioned population p.v. at first year
    - Output(All under `/[main_directory in config.py]/data_for_draw_fig/`):
        - Mean estimation of cohort by gamma fit via MLE: `cohort_gamma_fit_mean.csv`
        - Mean estimation of cohort by the midpoint method: `cohort_midpoint_mean.csv`
        - Mean estimation of cohort by VAM with gamma fit via MLE or the midpoint method: `cohort_qp_mean.csv`
        - Mean estimation of population by gamma fit via MLE: `population_gamma_fit_mean.csv`
        - Mean estimation of population by the midpoint method: `population_midpoint_mean.csv`
        - Mean estimation of cohort by VAM with gamma fit via MLE or the midpoint method: `population_qp_mean.csv`

6. estimation_error.py
    - Compute the estimation error of Gamma fit, midpoint method, QP(with midpoint method) and QP(with Gamma fit)
    - Includes mean estimation error and probability vector estimation error
    - Output: 
        - Estimation error of mean and p.v.: `/[main_directory in config.py]/estimation_error.csv`

### Pre-generated dataset

There is a pre-generated simulation datset using parameters shown in the article in `/simul_data/`

### Format of the generated dataset and results

1. Multi-year simulation data: `/[main_directory in config.py]/year_[number_of_year].csv`
    - Columns description: 
        
        | Column name          | Description                                                                              |
        |:-------------------- |:---------------------------------------------------------------------------------------- |
        | id                   | The identity of the observed individual                                                  |
        | sample               | The answer in the individual's mind                                                      |
        | q_result             | The survey response of the individual to the questionnaire that used in the current year |
        | new_q_result         | (Not in use) The individual answer the wrong answer                                      |
        | final_q_result       | (Not in use)    Merge the correct answer and wrong answer                                |
        | next_yr_ver_q_result | The survey response of the individual to the questionnaire that used in the next year    |
        
2. estimation error: `/[main_directory in config.py]/estimation_error.csv`
    - Columns description

        | Column name                   | Description                                                               |
        |:----------------------------- |:------------------------------------------------------------------------- |
        | error_cohort_gammafit_mean    | Estimation error by Gamma fit via MLE in the cohort data                  |
        | error_cohort_midpoint_mean    | Estimation error by the midpoint method in the cohort data                |
        | error_cohort_qp_midpoint_mean | Estimation error by VAM and the midpoint method in the cohort data        |
        | error_cohort_qp_gammafit_mean | Estimation error by VAM and Gamma fit via MLE in the cohort data          |
        | error_pop_gammafit_mean       | Estimation error by Gamma fit via MLE in the population samples           |
        | error_pop_midpoint_mean       | Estimation error by the midpoint method in the population samples         |
        | error_pop_qp_midpoint_mean    | Estimation error by VAM and the midpoint method in the population samples |
        | error_pop_qp_gammafit_mean    | Estimation error by VAM and Gamma fit via MLE in the population samples   |
        | linf_cohort_vec               | The probability vector estimation error of cohort in L infinity           |
        | kl_cohort_vec                 | The probability vector estimation error of cohort in KL divergence        |
        | linf_pop_vec                  | The probability vector estimation error of population in L infinity       |
        | kl_pop_vec                    | The probability vector estimation error of population in KL divergence    |


### Result of pre-generated dataset and default parameters

TODO: Some figs and tables?

<!-- ### Execution order and other tips

Please follow the order below to execute these programs.

1. add_randomness_simulation_data_gen.py
    - This code will generate simulation data, and store in the given directory.

2. add_randomness_simulation_prepare.py
    - This code will use the generated data to compute the essential matrices and vectors.
    - The essential matricse and vectors are store in the directory /qp_input_output/

3. Hand craft
    - Hand craft the ideal revision matrix and put it in /qp_input_output/ and named it Python_G_matrix.csv

4. main_qp.m
    - Use Matlab to solve Quadratic Problem, estimat the revised probability vector and the matrix A.

5. add_randomness_simulation_performance.py
    - Compute the estimated mean of the output of Matlab QP.
    - Prepare data for ploting the result in a figure.

6. add_randomness_simulation_draw_fig.py
    - Draw estimation result in a figure. -->

## Details about estimation by VAM(Quadratic programming)
<!-- TODO: Equation不能在github當中正常顯示，所以需要另尋他法 -->
Try to solve $B*A*c=f$ and $\epsilon * \theta_1 + \gamma * \theta_2$
The estimation of matrices and vectors is done by the main_qp.m script. 
They are all written in matlab scripts. 

### Input

What's the data format?

What's the input data and directory structure?

All the matrix below will compute by add_..._prepare.py, except Matrix G. (Please refer to the article for the method of computing matrix G manually)

1. Matrix M: Second year underlying distribution matrix
2. Matrix G: First year ideal revision matrix(need to be compute manually)
3. Matrix T: First year transition matrix(including underlying distribution change and revision)
4. Vector c: First year probability vector of response from cohort
5. Vector f: Second year probability vector of response from cohort

### Output

## Figures in the article

All the code and data that needed for drawing figures are in /article_figures/
