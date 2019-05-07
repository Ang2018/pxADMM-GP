function meanrBCM = rBCMpred(hyp, covfunc, localdata, test_x)

% rBCM prediction
%
%    hyp = [ log(ell_1)
%            log(ell_2)
%             .
%            log(ell_D)
%            log(sf) ]
%            log(sn) ]         , trained GP hyperparameters,
% where ell_1^2,...,ell_D^2 are ARD parameters, sf^2 is the signal
% variance, and sn^2 is the noise variance.

% Initialization
n = size(test_x, 1);
mu = zeros(n, 128);
cov = zeros(n, 128);

spmd (32)

    for kk = 1:128   
        [mu(:, kk), cov(:, kk)] = getPred(hyp, covfunc, ...
            localdata.x(1+170*(kk-1):170+170*(kk-1), :), ...
            localdata.y(1+170*(kk-1):170+170*(kk-1), :), test_x);      
    end
    
    beta = hyp(end-1) - 0.5 * log(cov);

    covrBCMinv = gop(@plus, sum(beta ./ cov, 2)) + ...
        (1 - gop(@plus, sum(beta, 2))) / exp(2 * hyp(end-1)) ;
    
    meanrBCM = gop(@plus, sum(beta .* mu ./ cov, 2)) ./ covrBCMinv;
    
end

meanrBCM = meanrBCM{1};

end
