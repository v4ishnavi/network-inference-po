function [Ahat, beta_hat] = graph_learning_si_LS(X, del_t, beta_init)
% SI graph learning using least squares with vectorization
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
T = size(X, 2);
N_process = size(X, 3);

% Compute numerical derivatives
dX = diff(X, 1, 2) / del_t;  % N x (T-1) x N_process
X_mid = 0.5 * (X(:, 1:T-1, :) + X(:, 2:T, :));  % midpoint rule

% Stack all data
v = [];  % derivatives
F = [];  % regression matrix

for proc = 1:N_process
    for t = 1:T-1
        x_t = X_mid(:, t, proc);
        dx_t = dX(:, t, proc);
        
        % Build F_t matrix: F_t * vec(A) = (I - diag(x_t)) * A * x_t
        % Using: A * x_t = (I ⊗ x_t') * vec(A)
        F_t = (eye(N) - diag(x_t)) * kron(eye(N), x_t');
        
        v = [v; dx_t];
        F = [F; F_t];
    end
end

% Solve: v = beta * F * vec(A)
% Use least squares
if size(F, 1) >= N^2 && rank(F) >= N^2
    a_vec = pinv(F) * (v / beta_init);
    Ahat = reshape(a_vec, N, N);
    
    % Enforce constraints
    Ahat = max(0, Ahat);  % non-negativity
    Ahat = Ahat - diag(diag(Ahat));  % no self-loops
    
    beta_hat = beta_init;
else
    Ahat = zeros(N, N);
    beta_hat = beta_init;
end

end
