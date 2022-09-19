# GroupedDataWithSurveyRevision

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

Reminder: R is installed with rpy2(a python package list above), but the package list below need to be installed manually
- Nloptr

### Matlab

- R2020a
- Optimization toolbox version 8.5

## Usage
<!-- TODO: 整理成一個bat檔案? -->
Please follow the execution order below to reproduce the result that we report in the article. 

Please note that the R code is integrate in the python code, so there is no need to execute R code in a separate R console. On the contrary, the Matlab code(main_qp.m) needs to be executed in a separate Matlab console, since it's code and interpreter are not integrated in the python. 

1. Check parameters in the `config_version/config.py`. 

<!-- TODO: 執行完後會給你甚麼? e.g. simulation dataset -->
2. Execute `config_version/generate_simul_data.py`

<!-- TODO: 加入理想改版矩陣的格式(.csv內的格式) -->
3. Hand craft the ideal revision matrix name it `Python_G_matrix.csv` and put it in `config_version/simul_data/qp_input_output/`. The format of `Python_G_matrix.csv` please refer to the pre-generated dataset. 

4. Execute `config_version/preprocessing_for_qp.py`

5. Execute `config_version/matlab_qp/main_qp.m` in a Matlab console, also the parameter of similarity can be edited in the code. The main_qp.m need a file path as a input parameter, this file path needs to contain `qp_input_output/{python_G_matrix.csv, python_c_vec, python_f_vec, python_T_matrix, python_M_matrix}`, so the default input parameter will be `config_version/simul_data/`

6. Execute `config_version/estimation.py`

7. Execute `config_version/estimation_error.py`

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
            - List in the `preprocess_for_qp_output.csv`(In the Root folder)

        2. Under `/[main_directory in config.py]/qp_input_output/`
            - Cohort p.v. in the first year: `python_vector_c.csv`
            - Cohort p.v. in the second year: `python_f_vec.csv`
            - Transition matrix of cohort at first year: `python_T_matrix.csv`
            - Time-related matrix of cohort at second year: `python_M_matrix.csv`

4. main_qp.m(In `config_version/matlab_qp/`)
    - Solve the Quadratice Programming problem
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
        - Mean estimation of the first cohort by VAM with gamma fit via MLE or the midpoint method: `cohort_qp_mean.csv`
        - Mean estimation of population by gamma fit via MLE: `population_gamma_fit_mean.csv`
        - Mean estimation of population by the midpoint method: `population_midpoint_mean.csv`
        - Mean estimation of population by VAM with gamma fit via MLE or the midpoint method: `population_qp_mean.csv`

6. estimation_error.py
    - Compute the estimation error of Gamma fit, midpoint method, QP(with midpoint method) and QP(with Gamma fit)
    - Includes mean estimation error and probability vector estimation error
    - Output: 
        - Estimation error of mean and p.v.: `/[main_directory in config.py]/estimation_error.csv`

### Pre-generated dataset

There is a pre-generated simulation datset in `config_version/simul_data/`, which is generated by using the parameters shown in the article

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

- The default parameter of the pre-generated dataset is listed in the *Simulation* section of the article
- The experiment result of the pre-generated dataset is in the *Result* section of the article
- The output files that mentioned in the *code* section above are all pre-generated in the default place
- The result of simulation in the article is using parameter of proportion below

    | $\gamma$ | $\epsilon$ | $\beta$ | $\alpha$ |
    |:--------:|:----------:|:-------:|:--------:|
    |    8     |     8      | 1.0706  |    1     |


## Details about estimation by VAM(Quadratic programming)
<!-- TODO: Equation不能在github當中正常顯示，所以需要另尋他法 -->
Try to solve the optimization problem below. 
$$
min\ \gamma \theta_1 + \epsilon \theta_2 + \beta \theta_3 + \alpha \theta_4
$$
subject to
$$
\sum^{n}_{k=1}(B)_{kj}=1,\forall j=1,...,m \\
\sum^{n}_{k=1}(A)_{kj}=1, \forall j=1,...,m\\
|B_1*A_1-T_1|_\infty<\gamma\\
|B_1*A_1*U_1-V'_1|_\infty<\epsilon\\
|B_1-B_2|_\infty<\beta\\
|A_1-G_1|_\infty<\alpha
$$
The estimation of matrices and vectors is done by the main_qp.m script. 
They are all matlab scripts. 

### Parameter of similarity
According to the article, tuning the parameter of proportion to get desire variable of similarity is important.  

The place that can change the parameter is at the line 60 in main_qp.m, 
the last four numbers are $\theta_1$ to $\theta_4$, respectively. 
```
f = [zeros(45,1);5;5;4;1];
```

### Input

All the matrices and vectors below will be computed by preprocessing_for_qp.py, except Matrix G. (Please refer to the *Algorithm VAM* section in the article for the method of computing matrix G manually)

The main_qp.m need one argument which is the path of the directory that includes `config_version/simul_data/qp_input_output/{python_G_matrix.csv, python_c_vec, python_f_vec, python_T_matrix, python_M_matrix}`，and these files will be treated as the matrices and vectors inputs below

1. Matrix M: Second year underlying distribution matrix
2. Matrix G: First year ideal revision matrix(need to be compute manually)
3. Matrix T: First year transition matrix(including underlying distribution change and revision)
4. Vector c: First year probability vector of response from cohort
5. Vector f: Second year probability vector of response from cohort

### Output
Listed in the *Simulation code and dataset* section above. (Same as the output of the main_qp.m)

## Bootstrap analysis
```
Note that the program in this part may take some time to execute.
```

In the article there is a bootstrap analysis for verifying the estimate probability vector is reasonable. The following jupyter notebook file(.ipynb) generates some data that can construct the 95% CI. The process of generating the data for bootstrap analysis is clearly mentioned in the article's *Result* section.  

### How to use 
1. Check the parameters in the `config_version/bootstrap/gamma_expectedValue_with_param_shift.py`, the sample size(`sample_size`) and number of samples(`sample_count`). Note that the result in the article used fixed number of samples=1000, and the sample size is set similar to the MJ dataset(Population: 57000; Cohort: 20000).  

2. Execute `config_version/bootstrap/gamma_expectedValue_with_param_shift.py`


3. The output is the `./gamma_shift_inc.csv`. Output with default parameter is already in the `config_version/bootstrap/`(`./gamma_shift_inc_20000.csv` and `./gamma_shift_inc_57000.csv`)

### Format of the output
The first row lists the second parameter of the first Gamma distribution, which is default drifted by 0.01(Can be set in the configuration parameter `gamma_shift_step`). And the other parameters are fixed, for example, first parameter of the first Gamma distribution, and 1st and 2nd parameter of the second Gamma distribution. 

The second and third row lists the lower and upper endpoint of the 95%CI computed in L infinity. The fourth and fifth row lists the lower and upper endpoint of the 97.5%CI computed in L infinity. The sixth row lists the statistics about the distribution of L infinity between the samples. 

The seventh to eleventh rows lists the same things as the second to sixth rows, but using KL divergence to compute the distance between probability of the samples. 
