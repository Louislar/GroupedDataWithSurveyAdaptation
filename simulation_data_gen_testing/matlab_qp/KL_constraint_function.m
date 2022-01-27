function [y,yeq,grady,gradyeq] = KL_constraint_function(x,M,T,G,c,f,J,p,q)
% y = 1/2*x'*Q*x + f'*x + c;

% data construct
% x = [b_11, ..., b_55, a_11, ..., a_54, epsilon, gamma, beta, alpha],
% 總共49個變數
% M: (5x5) matrix 98年的習慣矩陣
% T: (5x5) matrix 97年的總變異矩陣
% G: (5x4) matrix 97年的改版矩陣(使用gamma sampling重新估計出來的)

% 總共有幾個constraint的方程式
num_of_constraints = 14;
y = zeros(1,num_of_constraints);    % 總共14個constraint方程式的結果要輸出
% 輸出結果依序為: 
% B 約等於 M 5個
% BA 約等於 T 4個
% A 約等於 G 4個
% BAc=r 約等於 f 1個


% 計算KL-divergence版本的constraint
% B 大約等於 M
% 對應變數為 beta = x_48
% 每一行B和每一行M算KL divergence，總共有5行，也就是總共有5個方程式
result_of_KL_B_M = cell(5, 1);
B_matrix = reshape(x(1:25), [5, 5])';
for i = 1:5
    a_col_of_B = B_matrix(1+(i-1)*5:5+(i-1)*5);
    a_col_of_M = M(1+(i-1)*5:5+(i-1)*5);
    KL_result = KL_divergence(a_col_of_M, a_col_of_B);
    KL_result = KL_result - x(48);
    result_of_KL_B_M{i} = KL_result; 
end


% A 大約等於 G
% 每一行A和每一行G做cross entropy，總共有4行
% 對應變數為 alpha = x_49
result_of_KL_A_G = cell(4, 1);
A_matrix = reshape(x(26:45), [4, 5])';
for i = 1:4
    a_col_of_A = A_matrix(1+(i-1)*5:5+(i-1)*5);
    a_col_of_G = G(1+(i-1)*5:5+(i-1)*5);
    KL_result = KL_divergence(a_col_of_G, a_col_of_A);
    KL_result = KL_result - x(49);
    result_of_KL_A_G{i} = KL_result;
end


% BA 大約等於 T 
% 每一行BA和每一行T做cross entropy，總共有4行
sum_of_cross_entropy_BA_T = cell(4,1);
BA_matrix = B_matrix * A_matrix;
for i = 1:4
    a_col_of_BA = BA_matrix(1+(i-1)*5:5+(i-1)*5);
    a_col_of_T = T(1+(i-1)*5:5+(i-1)*5);
    KL_result = KL_divergence(a_col_of_T, a_col_of_BA);
    KL_result = KL_result - x(47);
    sum_of_cross_entropy_BA_T{i} = KL_result;
end

% BAc 大約等於 f
% vector的Cross entropy
BAc_vec = BA_matrix * c';
cross_entropy_BAc_f = cell(1,1);
KL_result = KL_divergence(f, BAc_vec);
KL_result = KL_result - x(46);
cross_entropy_BAc_f{1} = KL_result;

all_KL = [result_of_KL_B_M ; result_of_KL_A_G ; sum_of_cross_entropy_BA_T ; cross_entropy_BAc_f];
all_KL = cell2mat(all_KL);
y=all_KL';


% 新增的等式constraint計算
kk = length(J); % kk is the number of equality constraints
yeq = zeros(1,kk);
for i = 1:kk
    yeq(i) = x'*J{i}*x + p{i}'*x + q{i};
end



% 如果caller要求兩個輸出的話，就要算gradient
if nargout > 1
    % 該方程式沒有使用到的變數就設為0
    grady = zeros(length(x),num_of_constraints);
    % B 大約等於 M的五個方程式對所有變數的gradient
    % 有使用到的變數才填入gradient，其它不填的自動為0
    % 前五個方程式對beta的gradient永遠為-1
    grady(48,1:5)=-1;
    % 每個loop處理一個方程式的gradient
    % 每個方程式依序對應到每個B、M的column
    for i = 1:5
        a_col_of_B = B_matrix(1+(i-1)*5:5+(i-1)*5);
        a_col_of_M = M(1+(i-1)*5:5+(i-1)*5);
        after_divide = -a_col_of_M./a_col_of_B;
        grady(1+i-1:5:21+i-1,i) = after_divide; 
    end
    
    % A 大約等於 G的4個方程式對所有變數的gradient
    % 方程式編號從6開始，並且有4個方程式
    % alpha的gradient都是-1
    grady(49,6:9)=-1;
    for i = 1:4
        a_col_of_A = A_matrix(1+(i-1)*5:5+(i-1)*5);
        a_col_of_G = G(1+(i-1)*5:5+(i-1)*5);
        after_divide = -a_col_of_G./a_col_of_A;
        grady(26+i-1:4:42+i-1,i+5) = after_divide;
    end
    
    % BA 大約等於 T的4個方程式對所有變數的gradient
    % 方程式編號從10開始，並且有4個方程式
    % gamma的gradient都是-1
    grady(47,10:13)=-1;
    for i = 1:4 % 4個方程式
        a_col_of_T = T(1+(i-1)*5:5+(i-1)*5);
        a_col_of_A = A_matrix(1+(i-1)*5:5+(i-1)*5);
        
        BA_value = zeros(5, 1);
        for j=1:5
            a_row_of_B = B_matrix(j, 1:5);
            after_dot_product = a_row_of_B*a_col_of_A';
            BA_value(j) = after_dot_product;
        end
   
        
        gradient_of_an_A_col = zeros(1, 5); % A的第i個column的gradient
        for j=1:5
            a_col_of_B = B_matrix(1+(j-1)*5:5+(j-1)*5);
            after_dot_product = a_col_of_T.*a_col_of_B;
            after_divide = -after_dot_product./BA_value';
            gradient_of_an_A_col(j) = sum(after_divide);
        end
        
        % B的gradient重新計算(之前的微分是錯的)
        
        gradient_of_B = zeros(5, 5);    % 整個B矩陣的gradient
        for j=1:5
            after_dot_product = a_col_of_T(j).*a_col_of_A;
            gradient_of_an_b_row = -after_dot_product./BA_value(j);
            gradient_of_B(j, 1:5) = gradient_of_an_b_row'; 
        end
        grady(26+i-1:4:42+i-1, i+9) = gradient_of_an_A_col;
        grady(1:25, i+9) = reshape(gradient_of_B', [25, 1]);
    end
    
    % BAc 大約等於 f的4個方程式對所有變數的gradient
    % 方程式編號是14
    % epsilon的gradient是-1
    grady(46,14) = -1;
    % 分成5個數值計算 (與f的第幾個元素相乘) 
    % 注意，最後還是屬於同一個方程式的gradient，所以要加總回grady中的同一個column
    for i = 1:5
        a_row_of_B = B_matrix(i, 1:5);
        BAc_value=zeros(4,1);
        for j = 1:4
            a_col_of_A = A_matrix(1:5, j);
            a_cell_of_c = c(j);
            BA = a_row_of_B * a_col_of_A;
            BAc_value(j) = BA*a_cell_of_c;
        end
        
        % A的gradient
        gradient_of_A = zeros(5,4);
        for j=1:5
            after_dot_product = a_row_of_B(j)*c;
            after_dot_product = -f(i).*after_dot_product;
            after_divide = after_dot_product./BAc_value';
            gradient_of_A(j,1:4)=after_divide;
        end
        
        % B的第i個row的gradient
        gradient_of_an_B_row = zeros(1,5);
        for j=1:5
            after_dot_product = A_matrix(j, 1:4)*c';
            after_dot_product = -f(i)*after_dot_product;
            after_divide = after_dot_product./BAc_value;
            gradient_of_an_B_row(j) = sum(after_divide);
        end
        grady(26:45,14) = grady(26:45,14) + reshape(gradient_of_A', [20,1]);
        grady(1+(i-1)*5:5+(i-1)*5,14) = grady(1+(i-1)*5:5+(i-1)*5,14) + gradient_of_an_B_row';
    end
    
    % 新增的等式constraint計算
    gradyeq = zeros(length(x),kk);
    for i = 1:kk
        gradyeq(:,i) = 2*J{i}*x + p{i};
    end
    
end