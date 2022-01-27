% 使用quadratic programming計算最佳解 --> 包括習慣矩陣B和改版矩陣A

% objective function
Q = zeros(49, 49);
f = [zeros(45,1);5;5;4;1];    % 原始隨興給的參數
%f = [zeros(45,1);6;6;4;1];    % 原始隨興給的參數
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
four_param = x(46:49);

% 輸出想要的數值
tmp_97 = [0.5101349  0.23249741 0.09830508 0.15906261];   % 總體1997年的資料
% tmp_97 = [0.28547368 0.20189474 0.14294737 0.36968421];
tmp_rescreen_97 = [0.45, 0.25, 0.12, 0.18]  % 重複健檢人口的1997年資料
% tmp_rescreen_97 = [0.45245 0.1709  0.1081  0.26855];
midpoint_98 = [0.5, 1.75, 3.5, 5.5, 6.5];
trans_G_arr = G_arr';
trans_T_arr = T_arr';
version_change_matrix = B_arr*A_arr; % 估計的總變異矩陣
tmp_97 = tmp_97';
tmp_97_to_98_vec = A_arr*tmp_97;    % 估計的97年人填98年問卷結果
tmp_97_to_98_mean = midpoint_98  * tmp_97_to_98_vec;    % 估計的97年人填98年問卷結果，且用midpoint 算出數值

tmp_rescreen_97 = tmp_rescreen_97';
tmp_rescreen_97_to_98_vec = A_arr*tmp_rescreen_97;
tmp_rescreen_97_to_98_mean = midpoint_98  * tmp_rescreen_97_to_98_vec;
