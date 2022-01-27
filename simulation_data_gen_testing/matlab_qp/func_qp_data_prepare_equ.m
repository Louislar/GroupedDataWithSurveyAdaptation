function [J_5, J_6, p_5, p_6, q_5, q_6] = func_qp_data_prepare_equ()
% data construct (等式的constraints)
% x = [b_11, ..., b_55, a_11, ..., a_54, epsilon, gamma, beta, alpha],
% 總共49個變數 
% 接續quadratic
% 等式的符號換成: J, p, q

% 第五個方程式: B矩陣(習慣矩陣)的每個行加總要等於一 (等式方程式)
% 因為是一次式，所以2次項全為0
J_5 = cell(5, 1);
for i = 1:5
    J_5{i} = zeros(49, 49);
end

% 1次方程式共5個
p_5 = cell(5, 1);
for i= 1:5
    tmp_arr = zeros(49, 1);
    position = 1+i-1:5:21+i-1;
    tmp_arr(position)=1;
    p_5{i}=tmp_arr;
end

% 常數項
q_5 = cell(5,1);
for i = 1:5
    q_5{i}=-1;
end

% 第六個方程式: A矩陣(改版矩陣)的每個行加總要等於一 (等式方程式)
% 因為是一次式，所以2次項全為0
J_6 = cell(4, 1);
for i = 1:4
    J_6{i} = zeros(49, 49);
end

% 1次方程式共4個 (因為改版矩陣只有四個行要加總等於1)
p_6 = cell(4, 1);
for i= 1:4
    tmp_arr = zeros(49, 1);
    position = 26+i-1:4:42+i-1;
    tmp_arr(position)=1;
    p_6{i}=tmp_arr;
end

% 常數項
q_6 = cell(4,1);
for i = 1:4
    q_6{i}=-1;
end
end