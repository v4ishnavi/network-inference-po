%% --- SINGLE DEBUGGING EXPERIMENT: SI PARTIAL OBSERVABILITY ---
clear; clc; close all;

% Add paths for your local structure
addpath(fullfile('..', 'src'));

%% --- GLOBAL CONFIGURATION ---
run_id = 4;
N_obs = 12;               
N_hidden = 3;             
N_total = N_obs + N_hidden;
K = N_hidden;             

beta = 0.5;               
del_t = 0.05;             
T_duration = 5.0;         
division_factor = 1000;   
             
N_proc = 1;               
seed = 500;              

fprintf('=== SI DEBUGGdebugING RUN ===\n');
%% --- STEP 1: GENERATE GROUND TRUTH & DYNAMICS ---
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:)); 

obs_idx = 1:N_obs;
hid_idx = N_obs+1:N_total;
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, hid_idx);

t_vec = 0:del_t:T_duration;
Nt = length(t_vec);
X_obs_all = zeros(N_obs, Nt, N_proc);
X_hid_true_all = zeros(N_hidden, Nt, N_proc); % <--- Added to store ground truth

fprintf('Generating %d SI processes...\n', N_proc);
for pp = 1:N_proc
    X_full = generate_si_dynamics(A_full, t_vec, beta, seed + pp, division_factor);
    X_obs_all(:,:,pp) = X_full(obs_idx, :);
    X_hid_true_all(:,:,pp) = X_full(hid_idx, :); % <--- Store True Hidden
end

disp(size(X_obs_all))

% --- Run All Estimators ---
lambda = 1e-6;
results = run_granger_estimators(X_obs_all, lambda);

% --- Display Results ---
disp('Estimator execution successful.');
disp('Available estimates:');
disp(fieldnames(results));

% Example: Access Granger score matrix
A_hat_granger = results.Granger;

% Plotting the Granger estimate
figure;
imagesc(A_hat_granger); 
colorbar; 
title('Granger Estimator (Adjacency Matrix)');
xlabel('Source Node'); ylabel('Target Node');


function est = run_granger_estimators(X, lambda)
% RUN_GRANGER_ESTIMATORS Computes multiple Granger-based connectivity estimates
% 
% Usage:
%   est = run_granger_estimators(X, lambda)
%
% Inputs:
%   X      : (N x T) matrix of observed time series (e.g., infection probabilities)
%   lambda : Regularization parameter (e.g., 1e-3). 
%            CRITICAL for SI dynamics to preventing division by zero.
%
% Output:
%   est    : Struct containing the estimated adjacency matrices:
%            est.R1        (Lag-1 Correlation)
%            est.R1_R3     (Lag-1 minus Lag-3)
%            est.Precision (Inverse Covariance / Partial Correlation)
%            est.Granger   (Standard Granger Causality A = R1 * inv(R0))

    if nargin < 2
        lambda = 1e-3; % Default regularization if not provided
    end

    [N, T] = size(X);

    % ---------------------------------------------------------
    % 1. Precompute Covariance/Moment Matrices (R0, R1, R3)
    % ---------------------------------------------------------
    
    % R0: Lag-0 Moment (E[x_t * x_t'])
    % Note: We use raw moments (X*X') matching the Python helper.
    % If your data is not centered, this assumes 0-intercept.
    R0 = (X * X') / T;

    % R1: Lag-1 Moment (E[x_{t+1} * x_t'])
    % Python equivalent: z1=z[:,2:tsize], z2=z[:,1:tsize-1]
    X_future = X(:, 2:end);
    X_past   = X(:, 1:end-1);
    R1 = (X_future * X_past') / (T - 1);

    % R3: Lag-3 Moment (E[x_{t+3} * x_t'])
    % Python equivalent: z1=z[:,4:tsize], z2=z[:,1:tsize-3]
    if T > 3
        X_fut3 = X(:, 4:end);
        X_pst3 = X(:, 1:end-3);
        R3 = (X_fut3 * X_pst3') / (T - 3);
    else
        warning('Time series too short for Lag-3. R3 set to zeros.');
        R3 = zeros(N, N);
    end

    % ---------------------------------------------------------
    % 2. Regularized Inversion (The Fix for SI Data)
    % ---------------------------------------------------------
    % We add lambda to the diagonal of R0 before inverting.
    % This handles the "flatline" problem where variance -> 0.
    R0_reg = R0 + lambda * eye(N);
    
    % robust inverse (similar to numpy.linalg.lstsq or inv)
    R0_inv = inv(R0_reg);

    % ---------------------------------------------------------
    % 3. Compute Estimators
    % ---------------------------------------------------------

    % Method 1: Lag-1 Correlation
    % Logic: High correlation at lag 1 implies connection.
    est.R1 = abs(R1);

    % Method 2: R1 - R3
    % Logic: Subtracting Lag-3 removes "indirect" slow correlations,
    % keeping "direct" fast correlations.
    est.R1_R3 = abs(R1 - R3);

    % Method 3: Precision Matrix (Inverse Covariance)
    % Logic: Zeros in inverse covariance imply conditional independence.
    est.Precision = abs(R0_inv);

    % Method 4: Granger Causality (VAR Estimator)
    % Formula: A_hat = R1 * inv(R0)
    % Logic: "Does past X help predict future X better than past X alone?"
    est.Granger = abs(R1 * R0_inv);

end
