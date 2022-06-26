function [y,yeq,grady,gradyeq] = quadconstr(x,H,k,d,J,p,q)
jj = length(H); % jj is the number of inequality constraints
kk = length(J); % kk is the number of equality constraints

y = zeros(1,jj);
for i = 1:jj
    %disp(i);
    y(i) = x'*H{i}*x + k{i}'*x + d{i};
end
% 新增的等式constraint計算
yeq = zeros(1,kk);
for i = 1:kk
    yeq(i) = x'*J{i}*x + p{i}'*x + q{i};
end
    
if nargout > 2
    grady = zeros(length(x),jj);
    for i = 1:jj
        grady(:,i) = 2*H{i}*x + k{i};
    end
    % 新增的等式constraint計算
    gradyeq = zeros(length(x),kk);
    for i = 1:kk
        gradyeq(:,i) = 2*J{i}*x + p{i};
    end
end
