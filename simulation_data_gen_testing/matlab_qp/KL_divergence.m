function [out] = KL_divergence(p, q)

out = cross_entropy(p, q) - pmf_entropy(p);

end