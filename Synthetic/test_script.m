% generate test data

% ell^2 = 2, sf^2 = 1, sn^2 = 0.1
% ell = sqrt(2), sf = 1, sn = sqrt(0.1)
% hyperparameters in log scale
hyp = [log(sqrt(2)); log(sqrt(1)); log(sqrt(0.1))];
% number of training data points
n = 32000;

x = (1:n)';
K = mySEard(hyp(1:end-1), x);
K = K + exp(2 * hyp(end)) * eye(n);
L = chol(K, 'lower');
u = randn(n, 1);
% zero mean
y = 0 + L * u;

save('x.mat', 'x');
save('y.mat', 'y');

% training

% 16 local machines with 2000/1000/500 data points on each
% rho = 500, Lip = 5000
% inihyp = [1.5; 1.5; 1.5]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xx = x(1:32000);
yy = y(1:32000);

ttproxADMM32 = zeros(10, 1);
zzproxADMM32 = zeros(10, 3);

ttclasADMM32 = zeros(10, 1);
zzclasADMM32 = zeros(10, 3);

ttGrad32 = zeros(10, 1);
zzGrad32 = zeros(10, 3);

for i = 1:10
    
    display(i);
    localdata = splitfunc(xx, yy, 16, 2000);
    
    [t, z, history] = proxADMM(500, 5000, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttproxADMM32(i) = t;
    zzproxADMM32(i, :) = z';
    historyproxADMM32(i) = history;
    
    [t, z, history] = clasADMM(500, 1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttclasADMM32(i) = t;
    zzclasADMM32(i, :) = z';
    historyclasADMM32(i) = history;
    
    [t, z, history] = graDes(1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttGrad32(i) = t;
    zzGrad32(i, :) = z';
    historyGrad32(i) = history;
    
end

save('ttproxADMM32.mat', 'ttproxADMM32');
save('zzproxADMM32.mat', 'zzproxADMM32');
save('ttclasADMM32.mat', 'ttclasADMM32');
save('zzclasADMM32.mat', 'zzclasADMM32');
save('ttGrad32.mat', 'ttGrad32');
save('zzGrad32.mat', 'zzGrad32');
save('historyproxADMM32.mat', 'historyproxADMM32');
save('historyclasADMM32.mat', 'historyclasADMM32');
save('historyGrad32.mat', 'historyGrad32');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xx = x(1:16000);
yy = y(1:16000);

ttproxADMM16 = zeros(10, 1);
zzproxADMM16 = zeros(10, 3);

ttclasADMM16 = zeros(10, 1);
zzclasADMM16 = zeros(10, 3);

ttGrad16 = zeros(10, 1);
zzGrad16 = zeros(10, 3);

for i = 1:10
    
    display(i);
    localdata = splitfunc(xx, yy, 16, 1000);
    
    [t, z, history] = proxADMM(500, 5000, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttproxADMM16(i) = t;
    zzproxADMM16(i, :) = z';
    historyproxADMM16(i) = history;
    
    [t, z, history] = clasADMM(500, 1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttclasADMM16(i) = t;
    zzclasADMM16(i, :) = z';
    historyclasADMM16(i) = history;
    
    [t, z, history] = graDes(1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttGrad16(i) = t;
    zzGrad16(i, :) = z';
    historyGrad16(i) = history;
    
end

save('ttproxADMM16.mat', 'ttproxADMM16');
save('zzproxADMM16.mat', 'zzproxADMM16');
save('ttclasADMM16.mat', 'ttclasADMM16');
save('zzclasADMM16.mat', 'zzclasADMM16');
save('ttGrad16.mat', 'ttGrad16');
save('zzGrad16.mat', 'zzGrad16');
save('historyproxADMM16.mat', 'historyproxADMM16');
save('historyclasADMM16.mat', 'historyclasADMM16');
save('historyGrad16.mat', 'historyGrad16');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


xx = x(1:8000);
yy = y(1:8000);

ttproxADMM8 = zeros(10, 1);
zzproxADMM8 = zeros(10, 3);

ttclasADMM8 = zeros(10, 1);
zzclasADMM8 = zeros(10, 3);

ttGrad8 = zeros(10, 1);
zzGrad8 = zeros(10, 3);

for i = 1:10
    
    display(i);
    localdata = splitfunc(xx, yy, 16, 500);
    
    [t, z, history] = proxADMM(500, 5000, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttproxADMM8(i) = t;
    zzproxADMM8(i, :) = z';
    historyproxADMM8(i) = history;
    
    [t, z, history] = clasADMM(500, 1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttclasADMM8(i) = t;
    zzclasADMM8(i, :) = z';
    historyclasADMM8(i) = history;
    
    [t, z, history] = graDes(1e-5, [1.5; 1.5; 1.5], @mySEard, localdata);
    ttGrad8(i) = t;
    zzGrad8(i, :) = z';
    historyGrad8(i) = history;
    
end

save('ttproxADMM8.mat', 'ttproxADMM8');
save('zzproxADMM8.mat', 'zzproxADMM8');
save('ttclasADMM8.mat', 'ttclasADMM8');
save('zzclasADMM8.mat', 'zzclasADMM8');
save('ttGrad8.mat', 'ttGrad8');
save('zzGrad8.mat', 'zzGrad8');
save('historyproxADMM8.mat', 'historyproxADMM8');
save('historyclasADMM8.mat', 'historyclasADMM8');
save('historyGrad8.mat', 'historyGrad8');