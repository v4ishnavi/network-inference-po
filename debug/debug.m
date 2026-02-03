%% --- SINGLE DEBUGGING EXPERIMENT: SI PARTIAL OBSERVABILITY ---
clear; clc; close all;

% Add paths for your local structure
addpath(fullfile('..', 'src'));

%% --- GLOBAL CONFIGURATION ---

run_id = 4;

% Network Dimensions
N_obs = 10;               % Observed nodes
N_hidden = 2;             % Hidden nodes
N_total = N_obs + N_hidden;
K = N_hidden;             % Latent rank (Typically 1-3) 

% Dynamics Parameters
beta = 0.5;               % Infection rate
del_t = 0.05;             % Sampling resolution 
T_duration = 5.0;         % Total time (Early-time transients are best) 
division_factor = 1000;   % High-res simulation factor

% Inference Settings
max_iter = 500;           % Max EM iterations
tol = 1e-5;               % Convergence tolerance
method = "abs";           % Residual method for NMF init
init_mode = 'ls';         % 'ls' (recommended) or 'random' 
lambda = 1e-6;            % Ridge penalty for numerical stability 

% Data Volume
N_proc = 8;               % Number of independent SI processes
seed = 5;              % Fixed seed for reproducibility

fprintf('=== SI DEBUGGING RUN ===\n');
fprintf('Settings: N_obs=%d, N_hid=%d, K=%d, P=%d, Init=%s\n', ...
    N_obs, N_hidden, K, N_proc, init_mode);

%% --- STEP 1: GENERATE GROUND TRUTH & DYNAMICS ---
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:)); % Normalized Ground Truth

% save original to results folder run_id
A_full_orig = A_full;
% Create results directory if it doesn't exist
results_dir = fullfile('..', 'results', sprintf('run_%d', run_id));
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

save(fullfile(results_dir, 'A_full_orig.mat'), 'A_full_orig');

% Extract partitions 
obs_idx = 1:N_obs;
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, N_obs+1:end);

% Generate Multiple SI Processes 
t_vec = 0:del_t:T_duration;
X_obs_all = zeros(N_obs, length(t_vec), N_proc);

fprintf('Generating %d SI processes...\n', N_proc);
for pp = 1:N_proc
    % Each process needs a different initial condition to increase rank 
    X_full = generate_si_dynamics(A_full, t_vec, beta, seed + pp, division_factor);
    X_obs_all(:,:,pp) = X_full(obs_idx, :);
end

%% --- STEP 2: RUN INFERENCE ---
% This calls your EM/Alternating script
fprintf('Starting PO_model1 iterations...\n');
[A_hat, W_hat, Z_hat, hist] = PO_model1( ...
    X_obs_all, beta, del_t, K, max_iter, tol, A_full);

% save A_hat to results folder run_id
A_hat_result = A_hat;
save(fullfile(results_dir, 'A_hat_result.mat'), 'A_hat_result');

%% --- STEP 3: ANALYZE RESULTS ---
final_err_A = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
final_err_W = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');

fprintf('\n--- Final Metrics ---\n');
fprintf('A_OO Relative Error: %.2f%%\n', final_err_A * 100);
fprintf('W Relative Error   : %.2f%%\n', final_err_W * 100);
fprintf('Iterations to converge: %d\n', length(hist.obj));

%% --- STEP 4: VISUALIZATION ---
figure('Position', [100, 100, 1200, 800]);

% A: Objective Descent    
subplot(2, 2, 1);
semilogy(hist.obj, 'b-o', 'LineWidth', 1.5);
grid on; xlabel('Iteration'); ylabel('Objective Value');
title('Objective Function Descent');

% B: Parameter Change 
subplot(2, 2, 2);
plot(hist.dA * 100, 'r-s', 'LineWidth', 1.5, 'DisplayName', 'A Error');
hold on;
plot(hist.dW * 100, 'g-^', 'LineWidth', 1.5, 'DisplayName', 'W Error');
grid on; xlabel('Iteration'); ylabel('Error vs Truth (%)');
title('Parameter Recovery Trajectory');
legend;

% C: Adjacency Comparison (True)
subplot(2, 2, 3);
imagesc(A_true_obs); colorbar; axis square;
title('True A_{OO}');

% D: Adjacency Comparison (Recovered) 
subplot(2, 2, 4);
imagesc(A_hat); colorbar; axis square;
title(sprintf('Recovered A_{OO} (Err: %.1f%%)', final_err_A*100));

sgtitle(sprintf('SI Partial Observability Debugging: N=%d, K=%d', N_total, K));