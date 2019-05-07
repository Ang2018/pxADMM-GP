function subset = splitfunc(x, y, N, K)

% Create training subsets with certain overlap between each other, every
% single date point falls into at least one subset.
%
% x: L*D matrix where D is the input dimension, y: L*1 array
%
% L: number of all training data points
% N: number of local machines(subsets)
% K: number of training data points on each local machine
%
% K * N >= L

if size(x, 1) ~= size(y, 1), error('Size not match'); end
if K * N < size(x, 1), error('Data loss'); end

L = size(x, 1);
% one subset for each local machine
subset = Composite(N);
idx = randperm(L);
splitter = 1 : floor(L / N) : L;


for i = 1 : N
    
    % split whole dataset into N parts, without overlap
    if i == N
        subidx = idx(splitter(i) : end);
        restidx = idx;
        restidx(splitter(i) : end) = [];
    else
        subidx = idx(splitter(i) : splitter(i + 1) - 1);
        restidx = idx;
        restidx(splitter(i) : splitter(i + 1) - 1) = [];
    end
    
    % create overlap
    if numel(subidx) < K
        subidx = [subidx datasample(restidx, K - numel(subidx), 'Replace', false)];
    end
    
    subset{i} = struct('x', x(subidx, :), 'y', y(subidx, :));
    
end


end

