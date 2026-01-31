function [Ahat, beta_hat] = graph_learning_si_LS2(X, del_t, beta_init)
% SI graph learning with symmetry constraint using half-vectorization
%
% Inputs:
%   X: data matrix (N x T x N_process)
%   del_t: time step
%   beta_init: initial guess for beta (optional)
%
% Outputs:
%   Ahat: estimated symmetric adjacency matrix
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
        
        % Build F_t matrix for symmetric A
        F_t = [];
        for i = 1:N
            F_t_row = zeros(1, N*(N+1)/2);
            idx = 1;
            
            factor_i = (1 - x_t(i));
            for j = 1:N
                for k = j:N 
                    if j == k
                        coeff = 0;
                    elseif i == j
                        % a_jk affects node j
                        coeff = factor_i * x_t(k);
                    elseif i == k
                        % a_jk = a_kj affects node k  
                        coeff = factor_i * x_t(j);
                    else
                        coeff = 0;
                    end
                    F_t_row(idx) = coeff;
                    idx = idx + 1;
                end
            end
            F_t = [F_t; F_t_row];
        end
        
        v = [v; dx_t];
        F = [F; F_t];
    end
end

% Add constraint that diagonal elements are zero
N_params = N*(N+1)/2;
F_diag = zeros(N, N_params);
v_diag = zeros(N, 1);

idx = 1;
for j = 1:N
    for k = j:N
        if j == k  % diagonal element
            F_diag(j, idx) = 1;
        end
        idx = idx + 1;
    end
end

% Combined system
F_total = [F; F_diag];
v_total = [v / beta_init; v_diag];

% Solve least squares
if size(F_total, 1) >= N_params && rank(F_total) >= N_params
    a_sym = pinv(F_total) * v_total;
    
    % Convert back to matrix form
    Ahat = zeros(N, N);
    idx = 1;
    for j = 1:N
        for k = j:N
            if j == k
                Ahat(j, k) = 0;  % diagonal
            else
                val = max(0, a_sym(idx));  % non-negativity
                Ahat(j, k) = val;
                Ahat(k, j) = val;  % symmetry
            end
            idx = idx + 1;
        end
    end
    
    beta_hat = beta_init;
else
    Ahat = zeros(N, N);
    beta_hat = beta_init;
end

end
