%% --- EXPERIMENT: N_obs Scaling vs Recovery Error ---
clear; clc; close all;

% Add paths
addpath(fullfile('..', 'src'));

%% --- CONFIGURATION ---
N_obs_range = 4:1:20;      % N_obs from 4 to 20 (step of 2 for speed, or use 4:20)
N_hidden = 2;              % Fixed latent nodes
num_seeds = 50;            % Number of independent trials per point
N_proc = 8;                % Number of SI processes per trial

% Dynamics Parameters
beta = 0.5;               
del_t = 0.05;             
T_duration = 5.0;         
division_factor = 1000;   

% Solver Parameters
max_iter = 100;            % Reduced for bulk experiment efficiency
tol = 1e-4;               

% Initialize storage
mean_err_A = zeros(length(N_obs_range), 1);
std_err_A  = zeros(length(N_obs_range), 1);
mean_err_W = zeros(length(N_obs_range), 1);
std_err_W  = zeros(length(N_obs_range), 1);

fprintf('Starting Experiment: N_obs scaling (K=%d, Seeds=%d)\n', N_hidden, num_seeds);
t_start = tic;

%% --- MAIN EXPERIMENTAL LOOP ---
for i = 1:length(N_obs_range)
    n_obs = N_obs_range(i);
    n_total = n_obs + N_hidden;
    
    % Temporary storage for seeds within this N_obs
    seeds_err_A = zeros(num_seeds, 1);
    seeds_err_W = zeros(num_seeds, 1);
    
    fprintf('Testing N_obs = %d: ', n_obs);
    
    % Use 'parfor' if you have Parallel Computing Toolbox, otherwise 'for'
    for s = 1:num_seeds
        current_seed = s * 1000 + n_obs; % Ensure unique seeds per configuration
        
        % 1. Generate Ground Truth
        [A_full, ~, ~] = generate_graph(n_total, current_seed);
        A_full = max(0, A_full);
        A_full = A_full ./ max(A_full(:)); 
        
        obs_idx = 1:n_obs;
        hid_idx = n_obs+1:n_total;
        A_true_obs = A_full(obs_idx, obs_idx);
        W_true = A_full(obs_idx, hid_idx);
        
        % 2. Generate Dynamics
        t_vec = 0:del_t:T_duration;
        Nt = length(t_vec);
        X_obs_all = zeros(n_obs, Nt, N_proc);
        
        for pp = 1:N_proc
            X_full = generate_si_dynamics(A_full, t_vec, beta, current_seed + pp, division_factor);
            X_obs_all(:,:,pp) = X_full(obs_idx, :);
        end
        
        % 3. Inference (Suppressing output for bulk run)
        % Note: I added a 'silent' mode to my mental model of your function, 
        % but since PO_model2 prints, expect some terminal scroll.
        [A_hat, W_hat, ~, ~] = PO_model2(X_obs_all, beta, del_t, N_hidden, max_iter, tol, A_full);
        
        % 4. Store Errors
        seeds_err_A(s) = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
        seeds_err_W(s) = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
    end
    
    % Calculate Statistics
    mean_err_A(i) = mean(seeds_err_A);
    std_err_A(i)  = std(seeds_err_A);
    mean_err_W(i) = mean(seeds_err_W);
    std_err_W(i)  = std(seeds_err_W);
    
    fprintf('Done. Mean Err A: %.2f%%, W: %.2f%%\n', mean_err_A(i)*100, mean_err_W(i)*100);
end

total_time = toc(t_start);
fprintf('\nExperiment Complete in %.2f minutes.\n', total_time/60);

%% --- VISUALIZATION ---
figure('Color', 'black', 'Position', [100, 100, 800, 500]);

% Plot A_OO error
errorbar(N_obs_range, mean_err_A * 100, std_err_A * 100, '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'DisplayName', 'A_{OO} (Observed-Observed)');
hold on;


grid on;
xlabel('Number of Observed Nodes (N_{obs})', 'FontSize', 12);
ylabel('Relative Recovery Error (%)', 'FontSize', 12);
title(sprintf('Network Recovery Error vs. Observability (K=%d, %d Seeds)', N_hidden, num_seeds), 'FontSize', 14);
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

% Optional: Save results
% save('nobs_scaling_results.mat', 'N_obs_range', 'mean_err_A', 'std_err_A', 'mean_err_W', 'std_err_W');