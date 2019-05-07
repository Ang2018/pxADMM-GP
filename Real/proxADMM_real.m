function [t, z, history] = proxADMM_real(rho, Lip, inihyp, covfunc, localdata)

% Distributed GP via proximal ADMM
%
% rho, Lip: ADMM penalty parameter and Lipschitz constant, can be
% different across local machines.
% N: number of local machines (data subsets)
% K: number of training data points on each local machine
% inihyp = [ log(ell_1)
%            log(ell_2)
%             .
%            log(ell_D)
%            log(sf) ]
%            log(sn) ]         , initial guess of hyperparameters,
% where ell_1^2,...,ell_D^2 are ARD parameters, sf^2 is the signal
% variance, and sn^2 is the noise variance.

t_start = tic;

% ADMM parameters
QUIET    = 1;
MAX_ITER = 2000;
TOL      = 1e-7;

% initialization
n = size(inihyp, 1);
z = inihyp;
theta = repmat(inihyp, 1, 128);
beta = zeros(n, 128);

nlml = zeros(1, 128);
grad = zeros(n, 128);


if ~QUIET
    fprintf('%3s\t%10s\t%10s\n', 'iter', 's norm', 'objective');
end


for k = 1:MAX_ITER
    
    % Single Program Multiple Data (SPDM), cf. Chapter 10 of Boyd11
    spmd (32)
        % z-update
        zold = z;
        z = gop(@plus, sum(theta, 2) + sum(beta, 2) / rho) / (32 * 128);
        
        % theta-update
        for kk = 1:128
            [nlml(kk), grad(:, kk)] = getNlmlGrad(z, covfunc, ...
                localdata.x(1+170*(kk-1):170+170*(kk-1), :), ...
                localdata.y(1+170*(kk-1):170+170*(kk-1), :));
        end
        theta = z - (grad + beta) / (rho + Lip);
        
        % beta-update
        beta = beta + rho * (theta - z);
        
        % diagnostics, reporting, termination checks
        s_norm = norm(z - zold); 
        obj = gop(@plus, sum(nlml));        
    end

    history.z(k,:) = (z{1})';
    history.objval(k) = obj{1};
    history.t(k) = toc(t_start);
    
    if ~QUIET
        display((z{1})');
        fprintf('%3d\t%10.4f\t%10.2f\n', k, s_norm{1}, obj{1});
    end
    
    if and(s_norm{1} < TOL, k > 1)
        break;
    end
    
end


z = z{1};

if ~QUIET
    toc(t_start);
end

t = toc(t_start);

end
