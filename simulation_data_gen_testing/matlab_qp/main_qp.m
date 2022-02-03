function [version_change_matrix, habbit_matrix, new_pop_97_vec, new_cohort_97_vec] = main_qp(main_dir)
%{
Input: 
:M_arr: 98到99年的習慣矩陣
:G_arr: 97到98年預先估計的改版矩陣
:T_arr: 97到98年的總變異矩陣
:c_vec: 97年的人數比例向量
:f_vec: 98年的人數比例向量

output可能要輸出成csv會比較方便python讀取
Output: 
:ver_change_matrix: 估計出來的改版矩陣
:habbit_matrix: 估計出來的習慣矩陣
:new_pop_97_vec: 估計出來的總體97年人口比例 (考慮刪掉，留到python端做)
:new_cohort_97_vec: 估計出來的cohort 97人口比例
%}

% main_dir = '../mj_gamma_study/simulation_code/simulation_with_random_switch/';

% 處理 G matrix
input_G_data = readtable(strcat(main_dir, 'qp_input_output/python_G_matrix.csv')); 
G_data_size = size(input_G_data);
G_matrix = input_G_data.Variables;
G_matrix = G_matrix(2:G_data_size(1), 1:G_data_size(2)-1);
%disp(G_matrix)

% 處理 c vector
input_c_data = readtable(strcat(main_dir, 'qp_input_output/python_c_vec.csv')); 
c_vec = input_c_data.Variables;
%disp(c_vec);

% 處理 f vector
input_f_data = readtable(strcat(main_dir, 'qp_input_output/python_f_vec.csv')); 
f_vec = input_f_data.Variables; 
%disp(f_vec);

% 處理 T matrix
input_T_data = readtable(strcat(main_dir, 'qp_input_output/python_T_matrix.csv')); 
T_data_size = size(input_T_data);
T_matrix = input_T_data.Variables;
T_matrix = T_matrix(2:T_data_size(1), 1:T_data_size(2)-1 );
%disp(T_matrix);

% 處理 M matrix
input_M_data = readtable(strcat(main_dir, 'qp_input_output/python_M_matrix.csv')); 
M_data_size = size(input_M_data );
M_matrix = input_M_data.Variables;
M_matrix = M_matrix(2:M_data_size(1), 1:M_data_size(2)-1 );
%disp(M_matrix);

% 整理出方程式中的矩陣
[H_1, H_2, k_1, k_2, d_1, d_2] = func_qp_data_prepare_linear(M_matrix', G_matrix');
[H_3, H_4, k_3, k_4, d_3, d_4] = func_qp_data_prepare_quadratic(T_matrix', c_vec', f_vec');
[J_5, J_6, p_5, p_6, q_5, q_6] = func_qp_data_prepare_equ();



% objective function
Q = zeros(49, 49);
f = [zeros(45,1);5;5;4;1];    % 原始隨興給的參數
f = [zeros(45,1);6;6;4;1];    % 原始隨興給的參數
f = [zeros(45,1);7;7;4;1];    % 原始隨興給的參數
f = [zeros(45,1);8;8;4;1];    % 原始隨興給的參數
f = [zeros(45,1);8;8;7.4;1];    % 原始隨興給的參數 % 目前最好
%f = [zeros(45,1);8;8;1;1];
%f = [zeros(45,1);6;6;2.243011;1];
f=[zeros(45,1);8;8;1.0706;1];
%f = [zeros(45,1);7;7;1.608;1];
%f = [zeros(45,1);6;6;2.242913;1];
%f = [zeros(45,1);6;6;2.243075;1];
%f = [zeros(45,1);8;8;1.14258163;1];
%f = [zeros(45,1);8;8;2.13759;1];
%f = [zeros(45,1);8;8;2.235;1];
% f = [zeros(45,1);9;9;6;1];
% f = [zeros(45,1);10;10;1.5;1];    % 原始隨興給的參數
% f = [zeros(45,1);7.3;7;0.5;0.5];    % 使用greed search找出來的最佳參數
c = 0;

% constraint functions
H = [H_1; H_2; H_3; H_4];
k = [k_1; k_2; k_3; k_4];
d = [d_1; d_2; d_3; d_4];
J = [J_5; J_6];
p = [p_5; p_6];
q = [q_5; q_6];

% quadratic programming
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'HessianFcn',@(x,lambda)quadhess(x,lambda,Q,H,J),...
    'MaxIterations', 10^5, 'MaxFunctionEvaluations', 10^4);
% upper bound and lower bound of x
lb = zeros(49, 1);
ub = ones(49, 1)*1;

fun = @(x)quadobj(x,Q,f,c);
nonlconstr = @(x)quadconstr(x,H,k,d,J,p,q);
x0 = ones(49,1)*10^-3; % Column vector
[x,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],lb,ub,nonlconstr,options);

% reshape x (估計出來的四個參數，以及習慣矩陣和改版矩陣)
B_arr = x(1:25);
B_arr = reshape(B_arr,[5,5])' 
A_arr = x(26:45);
A_arr = reshape(A_arr,[4,5])'
four_param = x(46:49)

% 輸出想要的數值
% tmp_97 = [0.5101349  0.23249741 0.09830508 0.15906261];   % 總體1997年的資料
tmp_97 = [0.28547368 0.20189474 0.14294737 0.36968421];
% tmp_rescreen_97 = [0.45, 0.25, 0.12, 0.18]  % 重複健檢人口的1997年資料
% tmp_rescreen_97 = [0.45245 0.1709  0.1081  0.26855];
tmp_rescreen_97 = c_vec';
midpoint_98 = [0.5, 1.75, 3.5, 5.5, 6.5];
trans_G_arr = G_matrix';
trans_T_arr = T_matrix';
habbit_matrix = B_arr;  % 估計的習慣改變矩陣
version_change_matrix = A_arr; % 估計的改版矩陣
tmp_97 = tmp_97';
tmp_97_to_98_vec = A_arr*tmp_97;    % 估計的97年人填98年問卷結果
tmp_97_to_98_mean = midpoint_98  * tmp_97_to_98_vec;    % 估計的97年人填98年問卷結果，且用midpoint 算出數值
new_pop_97_vec = tmp_97_to_98_vec % 估計的97年人填98年問卷結果


tmp_rescreen_97 = tmp_rescreen_97';
tmp_rescreen_97_to_98_vec = version_change_matrix*tmp_rescreen_97;
tmp_rescreen_97_to_98_mean = midpoint_98  * tmp_rescreen_97_to_98_vec;
new_cohort_97_vec = tmp_rescreen_97_to_98_vec;

% 輸出成.csv檔案
writematrix(version_change_matrix, strcat(main_dir, 'qp_input_output/matlab_version_change_matrix.csv'));
writematrix(habbit_matrix, strcat(main_dir, 'qp_input_output/matlab_habbit_matrix.csv'));
writematrix(new_cohort_97_vec, strcat(main_dir, 'qp_input_output/matlab_new_cohort_97_vec.csv'));
writematrix(four_param, strcat(main_dir, 'qp_input_output/matlab_four_param.csv'));
end