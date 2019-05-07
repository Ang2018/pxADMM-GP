function [t, theta, history] = graDes(alpha, inihyp, covfunc, localdata)

% Gradient descent

t_start = tic;

% gradient descent parameters
QUIET    = 1;
MAX_ITER = 1e9;
TOL      = 1e-3;

% initialization
theta = inihyp;
N = length(localdata);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\n', 'iter', 's norm', 'objective');
end

for k = 1:MAX_ITER
    
    thetaold = theta;
    
    % reporting
    history.z(k,:) = theta';
    
    % distributed gradient calculation
    spmd (N)
        [nlZ, dnlZ] = getNlmlGrad(thetaold, covfunc, localdata.x, localdata.y);
        nlml = gop(@plus, nlZ);
        grad = gop(@plus, dnlZ);
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

