function [Ahat, beta_hat] = graph_learning_si_MT(X, del_t, beta_init)
% SI graph learning using Matrix-Tensor approach (node-by-node)
%
% Inputs:
%   X: data matrix (N x T x N_process)
%   del_t: time step
%   beta_init: initial guess for beta (optional)
%
% Outputs:
%   Ahat: estimated adjacency matrix
%   beta_hat: estimated infection rate

if nargin < 3
    beta_init = 1.0;
end

N = size(X, 1);
disp("[DEBUG] N (number of nodes):");
disp(N);
T = size(X, 2);
N_process = size(X, 3);

% Compute numerical derivatives
dX = diff(X, 1, 2) / del_t;  % N x (T-1) x N_process
X_mid = 0.5 * (X(:, 1:T-1, :) + X(:, 2:T, :));  % midpoint rule
disp("[DEBUG] dX size:");
disp(size(dX));
disp("[DEBUG] X_mid size:");
disp(size(X_mid));

Ahat = zeros(N, N);
beta_hat = beta_init;
% Check for NaN in input data
fprintf('NaN in X: %d\n', sum(isnan(X(:))));
fprintf('NaN in dX: %d\n', sum(isnan(dX(:))));
fprintf('NaN in X_mid: %d\n', sum(isnan(X_mid(:))));
fprintf('beta_hat: %f\n', beta_hat);

% Check data ranges
fprintf('X range: [%f, %f]\n', min(X(:)), max(X(:)));
% Solve for each row independently
for k = 1:N
    % Stack data across processes and time
    y_k = [];  % derivatives for node k
    G_k = [];  % regression matrix for node k
    
    for proc = 1:N_process
        for t = 1:T-1
            x_t = X_mid(:, t, proc);
            y_k = [y_k; dX(k, t, proc)];
            
            % SI dynamics: dx_k/dt = beta * (1 - x_k) * sum_j(a_kj * x_j)
            factor_k = (1 - x_t(k));
            G_k = [G_k; beta_hat * factor_k * x_t'];
        end
    end
    disp("[DEBUG] G_k size for node " + num2str(k) + ":");
    disp(size(G_k));
    disp("[DEBUG] y_k size for node " + num2str(k) + ":");
    disp(size(y_k));
    % disp(G_k)
    disp("rank of G_k: " + num2str(rank(G_k)));
    % Solve: y_k = G_k * a_k (where a_k is k-th row of A)
    if size(G_k, 1) >= N && rank(G_k) >= N
        a_k = pinv(G_k) * y_k;
        Ahat(k, :) = max(0, a_k');  % enforce non-negativity
        Ahat(k, k) = 0;  % no self-loops
    end
end

% Make symmetric
Ahat = 0.5 * (Ahat + Ahat');

end
