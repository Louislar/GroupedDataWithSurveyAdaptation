# GroupedDataWithSurveyAdaptation

[![hackmd-github-sync-badge](https://hackmd.io/deR1Z80sTfee2bE8WDjZ5Q/badge)](https://hackmd.io/deR1Z80sTfee2bE8WDjZ5Q)


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

- Nloptr

### Matlab

- R2020a
- Optimization toolbox version 8.5

## Usage
<!-- TODO: 整理成一個bat檔案? -->
Please follow the execution order below to reproduce the result that we report in the article. 

1. Check parameters in the config.py. 

2. Execute generate_simul_data.py

<!-- TODO: 加入理想改版矩陣的格式(.csv內的格式) -->
3. Hand craft the ideal revision matrix name it `Python_G_matrix.csv` and put it in `/qp_input_output/`. The format of `Python_G_matrix.csv` please refer to the pre-generated dataset. 

4. Execute main_qp.m 

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
        - Under `/[main_directory in config.py]/matrix_and_vector/`
            - Population vector in first year: `first_year_population_vec.csv`
            - Population vector in first year but answering the questionnaire in second year: `first_year_population_second_year_q_vec.csv` 
        - TODO: 慢慢補上來
        - Under `/[main_directory in config.py]/qp_input_output/`
            - Cohort p.v. in the first year: `python_vector_c.csv`

4. main_qp.m
    - Solve the Quadratice Programming
    - Output: 

5. estimation.py
    - 

### Pre-generated dataset

There is a pre-generated simulation datset using parameters shown in the article in `/simul_data/`

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

Try to solve $B*A*c=f$ and $\epsilon * \theta_1 + \gamma * \theta_2$
The estimation of matrices and vectors is done by the main_qp.m script. 
They are all written in matlab scripts. 

### Input

What's the data format?

What's the input data and directary structure?

All the matrix below will compute by add_..._prepare.py, except Matrix G. (Please refer to the article for the method of computing matrix G)

1. Matrix M: Second year underlying distribution matrix
2. Matrix G: First year ideal revision matrix(need to be compute manually)
3. Matrix T: First year transition matrix(including underlying distribution change and revision)
4. Vector c: First year probability vector of response from cohort
5. Vector f: Second year probability vector of response from cohort

### Output

## Figures in the article

All the code and data that needed for drawing figures are in /article_figures/
