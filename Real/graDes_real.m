function [t, theta, history] = graDes_real(alpha, inihyp, covfunc, localdata)

% Gradient descent

t_start = tic;

% gradient descent parameters
QUIET    = 1;
MAX_ITER = 2000;
TOL      = 1e-7;

% initialization
n = size(inihyp, 1);
theta = inihyp;

nlZ = zeros(1, 128);
dnlZ = zeros(n, 128);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\n', 'iter', 's norm', 'objective');
end

for k = 1:MAX_ITER
    
    thetaold = theta;
    
    % reporting
    history.z(k,:) = theta';
    
    % distributed gradient calculation
    spmd (32)
        for kk = 1:128
            [nlZ(kk), dnlZ(:, kk)] = getNlmlGrad(thetaold, covfunc, ...
                localdata.x(1+170*(kk-1):170+170*(kk-1), :), ...
                localdata.y(1+170*(kk-1):170+170*(kk-1), :));
        end
        nlml = gop(@plus, sum(nlZ));
        grad = gop(@plus, sum(dnlZ, 2));
    end
    obj = nlml{1};
    allgrad = grad{1};
    
    % take step
    theta = thetaold - alpha * allgrad;
    
    % reporting
    history.objval(k) = obj;
    history.t(k) = toc(t_start);
    
    % termination check
    s_norm = norm(theta - thetaold);
    
    if ~QUIET
        display(theta');
        fprintf('%3d\t%10.4f\t%10.2f\n', k, s_norm, obj);
    end
    
    if s_norm < TOL
        break;
    end
    
end

if ~QUIET
    toc(t_start);
end

t = toc(t_start);

end

