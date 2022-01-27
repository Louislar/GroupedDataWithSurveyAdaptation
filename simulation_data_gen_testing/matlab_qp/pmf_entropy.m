function [out] = pmf_entropy(p)
number_of_element = length(p);

out = 0;
for i = 1:number_of_element
    if p(i) ~= 0
        out = out - sum(p(i)*log(p(i)));
    end
end

end