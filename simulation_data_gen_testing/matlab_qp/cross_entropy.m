function [out] = cross_entropy(p, q)
number_of_element = length(p);  % length of the PMF vector = length of the r.v. set
out = 0;
% cross entropy p to q
for i = 1:number_of_element
    out = out + p(i) * log(q(i));
end
out = out * (-1);
end