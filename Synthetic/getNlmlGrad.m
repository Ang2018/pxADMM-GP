function [nlZ, dnlZ] = getNlmlGrad(hyp, covfunc, x, y)

% nlZ is the negative log marginal likelihood and dnlZ its partial
% derivatives wrt the hyperparameters.

n = size(x, 1);
K = covfunc(hyp(1:end-1), x);
K = K + exp(2 * hyp(end)) * eye(n);
L = chol(K, 'lower');
alpha = L' \ (L \ y);
nlZ = -0.5 * y' * alpha - trace(log(L)) - n/2 * log(2 * pi);
% negative log marginal likelihood
nlZ = -nlZ;

invL = inv(L);
alphaalphainvK = alpha * alpha' - invL' * invL;
dnlZ = zeros(size(hyp));

for i = 1:numel(hyp(1:end-1))
    dKdtheta = covfunc(hyp(1:end-1), x, [], i);
    dnlZ(i) = 0.5 * sum(sum(alphaalphainvK .* dKdtheta'));
end

dnlZ(end) = 0.5 * sum(sum(alphaalphainvK .* (2 * exp(2 * hyp(end)) * eye(n))'));
% gradient of nlml
dnlZ = -dnlZ;

end

