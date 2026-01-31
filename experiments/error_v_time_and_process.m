% run_scaling_analysis.m
% Analyzes the performance of Partial Observability inference:
% a) Error vs Number of Samples (for 1 process)
% b) Error vs Number of Processes (for fixed time samples)

clear; clc; close all;

% Add bin and src folders to path
addpath(fullfile('..', 'bin'));
addpath(fullfile('..', 'src'));

%% --- GLOBAL CONFIGURATION ---
N_obs = 10;               % number of observed nodes
N_hidden = 2;             % number of hidden nodes
N_total = N_obs + N_hidden;
K = N_hidden;             % latent rank
beta = 0.5;               % infection rate (slower spread for better resolution)
del_t = 0.05;              % coarser time step
seed = 1000;                % Fixed seed for graph generation
max_iter = 400;
division_factor = 1000; % For dynamics generation
tol = 1e-5;
method = "abs";           % Residual method
init_mode = 'ls';         % Use LS initialization for consistent results

% Parameters for Exp A
T_max_duration = 5.0;     % Long enough to capture saturation
% Parameters for Exp B
T_fixed_duration = 5.0;    % Fixed duration (standard experiment length)

fprintf('=== SI INFERENCE SCALING ANALYSIS ===\n');
fprintf('N_obs=%d, N_hidden=%d, K=%d\n', N_obs, N_hidden, K);

% --- GENERATE GROUND TRUTH NETWORK (ONCE) ---
fprintf('Generating ground truth graph...\n');
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:)); % Normalize

% Extract true components for error calculation
obs_idx = 1:N_obs;
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, N_obs+1:end);

%% --- EXPERIMENT A: Error vs Number of Samples (T) for 1 Process ---
fprintf('\n--- Experiment A: Varying Time Samples (T) ---\n');

% Parameters for Exp A
% T_max_duration = 4.0;     % Long enough to capture saturation
t_full = 0:del_t:T_max_duration;
N_proc_fixed = 1;          % Fixed at 1 process

% Generate ONE long process
X_full_A = generate_si_dynamics(A_full, t_full, beta, seed, division_factor);
X_obs_A_all = X_full_A(obs_idx, :, 1:N_proc_fixed); % (N_obs x T_max x 1)

% Define sample sizes to test (indices of time vector)
% We ensure we take at least a few samples
sample_counts = [50, 100, 200, 300, 400, 500]; 
err_A_vs_T = zeros(length(sample_counts), 1);
err_W_vs_T = zeros(length(sample_counts), 1);

for i = 1:length(sample_counts)
    n_samples = sample_counts(i);
    
    % Check if we have enough generated data
    if n_samples > size(X_obs_A_all, 2)
        warning('Requested more samples than generated. Clipping.');
        n_samples = size(X_obs_A_all, 2);
    end
    
    % Slice data: Take first n_samples
    X_curr = X_obs_A_all(:, 1:n_samples, :);
    
    fprintf('  Testing T = %d samples... ', n_samples);
    
    % Run Inference
    try
        [A_hat, W_hat, ~, ~] = PO_model1( ...
            X_curr, beta, del_t, K, max_iter, tol, method, A_full, init_mode);
        
        % Calculate Errors
        err_A_vs_T(i) = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
        err_W_vs_T(i) = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
        fprintf('Error A: %.4f\n', err_A_vs_T(i));
    catch ME
        fprintf('Failed: %s\n', ME.message);
        err_A_vs_T(i) = NaN;
        err_W_vs_T(i) = NaN;
    end
end


%% --- EXPERIMENT B: Error vs Number of Processes (P) for Fixed T ---
fprintf('\n--- Experiment B: Varying Processes (P) ---\n');

% Parameters for Exp B
% T_fixed_duration = 4.0;    % Fixed duration (standard experiment length)
t_fixed = 0:del_t:T_fixed_duration;
N_proc_max = 20;           % Max processes to simulate

% Generate MANY processes (pool)
X_full_B = zeros(N_total, length(t_fixed), N_proc_max);
for pp = 1:N_proc_max
    % Use different seeds for different processes
    X_full_B(:,:,pp) = generate_si_dynamics(A_full, t_fixed, beta, seed+pp, division_factor);
end
X_obs_B_all = X_full_B(obs_idx, :, :);

% Define process counts to test
proc_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10];
err_A_vs_P = zeros(length(proc_counts), 1);
err_W_vs_P = zeros(length(proc_counts), 1);

for i = 1:length(proc_counts)
    n_proc = proc_counts(i);
    
    % Slice data: Take first n_proc processes
    X_curr = X_obs_B_all(:, :, 1:n_proc);
    
    fprintf('  Testing P = %d processes... ', n_proc);
    
    % Run Inference
    try
        [A_hat, W_hat, ~, ~] = PO_model1( ...
            X_curr, beta, del_t, K, max_iter, tol, method, A_full, init_mode);
        
        % Calculate Errors
        err_A_vs_P(i) = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
        err_W_vs_P(i) = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
        fprintf('Error A: %.4f\n', err_A_vs_P(i));
    catch ME
        fprintf('Failed: %s\n', ME.message);
        err_A_vs_P(i) = NaN;
        err_W_vs_P(i) = NaN;
    end
end

%% --- VISUALIZATION ---
fprintf('\nGenerating plots...\n');
plots_dir = fullfile('..', 'plots');
if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

figure('Position', [100, 100, 1200, 500]);

% Plot A: Error vs Samples
subplot(1, 2, 1);
plot(sample_counts, err_A_vs_T * 100, '-o', 'LineWidth', 2, 'DisplayName', 'A_{OO} Error');
hold on;
% plot(sample_counts, err_W_vs_T * 100, '-s', 'LineWidth', 2, 'DisplayName', 'W Error');
% grid on;
xlabel('Number of Time Samples (T)');
ylabel('Relative Error (%)');
title(sprintf('Error vs. Samples (1 Process)'));
legend('Location', 'Best');

% Plot B: Error vs Processes
subplot(1, 2, 2);
plot(proc_counts, err_A_vs_P * 100, '-o', 'LineWidth', 2, 'DisplayName', 'A_{OO} Error');
hold on;
% plot(proc_counts, err_W_vs_P * 100, '-s', 'LineWidth', 2, 'DisplayName', 'W Error');
% grid on;
xlabel('Number of Processes (P)');
ylabel('Relative Error (%)');
title(sprintf('Error vs. Processes (T=%.1fs)', T_fixed_duration));
legend('Location', 'Best');

sgtitle(sprintf('Partial Observability Scaling: N_{obs}=%d, N_{hid}=%d, K=%d', ...
    N_obs, N_hidden, K));

% Save plot
plot_name = sprintf('scaling_analysis_Nobs%d.png', N_obs);
saveas(gcf, fullfile(plots_dir, plot_name));
fprintf('Plot saved to %s\n', fullfile(plots_dir, plot_name));

%% Save Data Results
results_dir = fullfile('..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
save_name = 'scaling_analysis_results.mat';
save(fullfile(results_dir, save_name), 'sample_counts', 'err_A_vs_T', 'err_W_vs_T', ...
    'proc_counts', 'err_A_vs_P', 'err_W_vs_P');
fprintf('Results saved to %s\n', fullfile(results_dir, save_name));