% X: normalized training input
% Y: normalized training output
% Ystd: standard deviation of oringinal training output
% XT: normalized test input
% YT: oringinal test output subtracted by the mean of oringinal training output

load('X.mat');
load('XT.mat');
load('Y.mat');
load('YT.mat');
load('Ystd.mat');

RMSEADMM = zeros(10,1);
RMSEGrad = zeros(10,1);
ttADMM = zeros(10,1);
ttGrad = zeros(10,1);
zzADMM = zeros(10,10);
zzGrad = zeros(10,10);

for i=1:10
    
    display(i);
    
    idx = randperm(700000);
    XX = X(idx(1:696320), :);
    YY = Y(idx(1:696320), :);
    localdata = splitfunc(XX, YY, 32, 21760);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [t, z, history] = proxADMM_real(50, 500, [1;4.8;1;1;1;3;1;1;1;1], @mySEard, localdata);
    
    ttADMM(i) = t;
    zzADMM(i, :) = z';
    historyADMM(i) = history;
    
    meanrBCM = rBCMpred(z, @mySEard, localdata, XT);
    RMSEADMM(i) = sqrt(sum((meanrBCM * Ystd - YT)' * (meanrBCM * Ystd - YT)) / 100000);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [t, theta, history] = graDes_real(1e-6, [1;4.8;1;1;1;3;1;1;1;1], @mySEard, localdata);
    
    ttGrad(i) = t;
    zzGrad(i, :) = theta';
    historyGrad(i) = history;
    
    meanrBCM = rBCMpred(theta, @mySEard, localdata, XT);
    RMSEGrad(i) = sqrt(sum((meanrBCM * Ystd - YT)' * (meanrBCM * Ystd - YT)) / 100000);
    
end

save('ttADMM.mat', 'ttADMM');
save('ttGrad.mat', 'ttGrad');
save('zzADMM.mat', 'zzADMM');
save('zzGrad.mat', 'zzGrad');
save('RMSEADMM.mat', 'RMSEADMM');
save('RMSEGrad.mat', 'RMSEGrad');
save('historyADMM.mat', 'historyADMM');
save('historyGrad.mat', 'historyGrad');