function hess = quadhess(x,lambda,Q,H,J)
hess = Q;
jj = length(H); % jj is the number of inequality constraints
kk = length(J); % kk is the number of equality constraints
for i = 1:jj
    hess = hess + lambda.ineqnonlin(i)*2*H{i};
end
% 新增的等式constraint計算
for i = 1:kk
    hess = hess + lambda.eqnonlin(i)*2*J{i};
end