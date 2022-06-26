function [H_3, H_4, k_3, k_4, d_3, d_4] = func_qp_data_prepare_quadratic(T_arr, c_4, f_arr)

% data construct
% x = [b_11, ..., b_55, a_11, ..., a_54, epsilon, gamma, beta, alpha],
% 總共49個變數 
% 注意!!2次方程式的H，都不是symmetric的，不過只要換成(H + H^T)/2，就可以變成symmetric並且結果與原來一樣 
% 接續linear

% 第三個方程式: BA 大約等於 T, 最小化誤差參數使用gamma (為quadratically方程式)
% 2次方程式共40個
H_3 = cell(40,1);
for i = 1:20
    tmp_arr = zeros(49,49);
    b_position = 1+fix((i-1)/4)*5:5+fix((i-1)/4)*5;
    a_position = 26+mod((i-1),4):4:42+mod((i-1),4);
    for j=1:length(b_position)
        tmp_arr(b_position(j),a_position(j)) = 1;
    end
    
    H_3{i} = tmp_arr;
    H_3{i+20} = tmp_arr * -1;
end
% 2次方程式要改成(H + H^T)/2
for i = 1:length(H_3)
    H_3{i} = (H_3{i} + H_3{i}')/2;
end

% 1次方程式 
k_3 = cell(40, 1);
for i = 1:20
    tmp_arr = zeros(49,1);
    tmp_arr(47) = -1;
    k_3{i} = tmp_arr;
    k_3{i+20} = tmp_arr;
end

% 常數項
% 注意!!不加transpose 是因為matlab的array indexing是先走完列在走行
% T_arr = [
%     0.686624073, 0.190024732, 0.072135202, 0.028854081, 0.022361913, ;
%     0.342056764, 0.374124585, 0.183560634, 0.061371176, 0.038886841, ;
%     0.161226897, 0.283523398, 0.313802595, 0.143924499, 0.097522611, ;
%     0.091542032, 0.124033007, 0.227694688, 0.224342445, 0.332387829;
%     ];
% T_arr = [
%     0.683280,  0.303979,  0.221092,  0.105753, ;
%     0.200796,  0.410181,  0.356614,  0.164588, ;
%     0.071168,  0.185196,  0.270120,  0.268293, ;
%     0.025196,  0.061732,  0.099445,  0.209831, ;
%     0.019560,  0.038912,  0.052729,  0.251536
%     ]';
neg_T_arr = T_arr * -1;
d_3 = cell(40,1);
for i = 1:20
    d_3{i} = neg_T_arr(i);
    d_3{i+20} = T_arr(i);
end


% 第四個方程式: BAc 大約等於 f, 最小化誤差參數使用epsilon(為quadratically方程式)
% 2次方程式共10個
H_4 = cell(10, 1);
% 1997年人口比例
% c_4 = [
%     0.450280729, 0.25177486, 0.117999165, 0.179945246
%     ];
% c_4 = [
%     0.45245, 0.1709,  0.1081,  0.26855
%     ];
for i = 1:5
    tmp_arr = zeros(49, 49);
    position_row = 1+(i-1)*5:5+(i-1)*5;
    %position_col = 26:29
    for j = 1:length(position_row)
        position_col = 26+(j-1)*4:29+(j-1)*4;
        tmp_arr(position_row(j), position_col) = c_4;
    end
    H_4{i} = tmp_arr;
    H_4{i+5} = tmp_arr*-1;
end
% 2次方程式要改成(H + H^T)/2
for i = 1:length(H_4)
    H_4{i} = (H_4{i} + H_4{i}')/2;
end

% 1次方程式
k_4 = cell(10, 1);
for i = 1:5
   tmp_arr = zeros(49,1) ;
   tmp_arr(46) = -1;
   k_4{i} = tmp_arr;
   k_4{i+5} = tmp_arr;
end

% 常數項
d_4 = cell(10, 1);
% f_arr = [0.430792075, 0.235534314, 0.156698065, 0.085796483, 0.091179064];% 1998年人口比例
% f_arr = [0.4134, 0.2437, 0.1651, 0.08905, 0.08875]; 
neg_f_arr = f_arr * -1;
for i = 1:5
    d_4{i} = neg_f_arr(i);
    d_4{i+5} = f_arr(i);
end



end