function [score_list, combinations_of_params] = greed_search()
param_theta_cell_arr = {[7, 7.1, 7.3, 7.4], [6.5, 7, 7.2, 7.5, 8], [0.4, 0.5, 0.55, 1], [0.3, 0.4, 0.5, 0.55]};
combinations_of_params = combvec(param_theta_cell_arr{:}); % 所有嘗試參數的組合
% argsort in matlab (ref: https://stackoverflow.com/questions/39484073/matlab-sort-vs-numpy-argsort-how-to-match-results)
% 準備訓練用資料
qp_data_prepare_linear
qp_data_prepare_quadratic
qp_data_prepare_equ

% 建立每一種參數組合 (ref: https://stackoverflow.com/questions/4165859/generate-all-possible-combinations-of-the-elements-of-some-vectors-cartesian-pr)
% (ref: https://stackoverflow.com/questions/7446946/how-to-generate-all-pairs-from-two-vectors-in-matlab-using-vectorised-code)

% loop: 每一種參數組合下
%       使用QP求最佳解(ref: qp_main) 

% ======= QP 方程式設定 =======
% objective function
Q = zeros(49, 49);
f = [zeros(45,1);5;5;4;1];
c = 0;

% constraint functions
H = [H_1; H_2; H_3; H_4];
k = [k_1; k_2; k_3; k_4];
d = [d_1; d_2; d_3; d_4];
J = J_5;
p = p_5;
q = q_5;

% quadratic programming
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'HessianFcn',@(x,lambda)quadhess(x,lambda,Q,H,J),...
    'MaxIterations', 10^5, 'MaxFunctionEvaluations', 10^4);
% upper bound and lower bound of x
lb = zeros(49, 1);
ub = ones(49, 1)*1;

nonlconstr = @(x)quadconstr(x,H,k,d,J,p,q);
x0 = ones(49,1)*10^-3; % Column vector

% ======= QP 方程式設定 (end) =======

score_list = []
for i = 1:length(combinations_of_params)
    cur_param = combinations_of_params(:, i);
    f(46:49) = cur_param;
    
    fun = @(x)quadobj(x,Q,f,c);
    [x,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],lb,ub,nonlconstr,options);
    
    score_list = [score_list sum(x(46:49))];
end
end