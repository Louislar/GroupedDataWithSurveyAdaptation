function [version_change_matrix, habbit_matrix, new_pop_97_vec, new_cohort_97_vec] = KL_main_qp(main_dir)

% 處理 G matrix
input_G_data = readtable(strcat(main_dir, 'qp_input_output/python_G_matrix.csv')); 
G_data_size = size(input_G_data);
G_matrix = input_G_data.Variables;
G_matrix = G_matrix(2:G_data_size(1), 1:G_data_size(2)-1);
% disp(G_matrix)

% 處理 c vector
input_c_data = readtable(strcat(main_dir, 'qp_input_output/python_c_vec.csv')); 
% disp(input_c_data);
c_vec = input_c_data.Variables;

% 處理 f vector
input_f_data = readtable(strcat(main_dir, 'qp_input_output/python_f_vec.csv')); 
f_vec = input_f_data.Variables; 

% 處理 T matrix
input_T_data = readtable(strcat(main_dir, 'qp_input_output/python_T_matrix.csv')); 
T_data_size = size(input_T_data);
T_matrix = input_T_data.Variables;
T_matrix = T_matrix(2:T_data_size(1), 1:T_data_size(2)-1 );

% 處理 M matrix
input_M_data = readtable(strcat(main_dir, 'qp_input_output/python_M_matrix.csv')); 
M_data_size = size(input_M_data );
M_matrix = input_M_data.Variables;
M_matrix = M_matrix(2:M_data_size(1), 1:M_data_size(2)-1 );

% objective function
Q = zeros(49, 49);
f = [zeros(45,1);5;5;4;1];    % 原始隨興給的參數
f = [zeros(45,1);6;6;4;1];    % 原始隨興給的參數
f = [zeros(45,1);7;7;4;1];    % 原始隨興給的參數
f = [zeros(45,1);8;8;4;1];    % 原始隨興給的參數
f = [zeros(45,1);8;8;7.4;1];    % 原始隨興給的參數 % 目前最好
% 把初始值改成之前求解出來的答案當成初始值代入
previous_A_mat = [0.967845937	0.072637844	0.043807745	0.009037801 
1.39E-07	0.86136051	0.422671461	1.31E-06 
3.03E-07	0.066001602	0.533519102	0.264936627 
0.032144356	2.14E-08	9.25E-07	0.449385975 
9.27E-06	2.18E-08	7.68E-07	0.276638286 
];
previous_B_mat = [0.70119494	0.274392598	0.136448022	0.094277669	0.085540544 
0.207101885	0.435969547	0.301469645	0.124218676	0.07743186 
0.062513382	0.191371282	0.357643163	0.282360951	0.15627041 
0.014385658	0.058286193	0.13357709	0.296162198	0.17460351 
0.014804135	0.039980379	0.070862081	0.202980507	0.506153675 
];
% f(26:45) = reshape(previous_A_mat', [20,1]);
% f(1:25) = reshape(previous_B_mat', [25,1]);
c = 0;

% 等式的constraints
[J_5, J_6, p_5, p_6, q_5, q_6] = func_qp_data_prepare_equ();
J = [J_5; J_6];
p = [p_5; p_6];
q = [q_5; q_6];

% quadratic programming (先不要給hessian看會不會跑很久，太久再補)
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'MaxIterations', 10^5, 'MaxFunctionEvaluations', 10^7,...
    'Display', 'notify-detailed');
% upper bound and lower bound of x
% 注意!! KL距離的上下界並不是0到1之間
lb = zeros(45, 1);
ub = ones(45, 1)*1;
lb = [lb ;0 ;0 ;0;0];
ub = [ub; Inf; Inf; Inf; Inf];

fun = @(x)quadobj(x,Q,f,c);
nonlconstr = @(x)KL_constraint_function(x,M_matrix,T_matrix,G_matrix,c_vec',f_vec,J,p,q);
x0 = ones(49,1)*10^-3; % Column vector
x0(26:45) = reshape(previous_A_mat', [20,1]);
x0(1:25) = reshape(previous_B_mat', [25,1]);
[x,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],lb,ub,nonlconstr,options);

% reshape x (估計出來的四個參數，以及習慣矩陣和改版矩陣)
B_arr = x(1:25);
B_arr = reshape(B_arr,[5,5])'; 
A_arr = x(26:45);
A_arr = reshape(A_arr,[4,5])';
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

end