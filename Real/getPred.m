function [mu, cov] = getPred(hyp, covfunc, x, y, test_x)

% GP prediction
n = size(x, 1);
K = covfunc(hyp(1:end-1), x);
K = K + exp(2 * hyp(end)) * eye(n);
L = chol(K, 'lower');
alpha = L' \ (L \ y);
Kstar = covfunc(hyp(1:end-1), test_x, x);
mu = Kstar * alpha;

v = L \ Kstar';
cov = covfunc(hyp(1:end-1), test_x, 'diag') - sum(v .* v, 1)';

end

