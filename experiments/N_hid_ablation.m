%% --- EXPERIMENT: N_hidden Ablation vs Recovery Error ---
clear; clc; close all;

% Add paths
addpath(fullfile('..', 'src'));

%% --- CONFIGURATION ---
N_obs = 15;                % FIXED number of observed nodes
N_hidden_range = 0:10;     % Ablation range for hidden nodes
num_seeds = 50;            % Number of independent trials (reduced slightly for speed)
N_proc = 8;                % Number of SI processes per trial

% Dynamics Parameters
beta = 0.5;               
del_t = 0.05;             
T_duration = 5.0;         
division_factor = 1000;   

% Solver Parameters
max_iter = 100;            
tol = 1e-4;               

% Initialize storage
mean_err_A = zeros(length(N_hidden_range), 1);
std_err_A  = zeros(length(N_hidden_range), 1);
mean_err_W = zeros(length(N_hidden_range), 1);
std_err_W  = zeros(length(N_hidden_range), 1);

fprintf('Starting Experiment: N_hidden ablation (N_obs=%d, Seeds=%d)\n', N_obs, num_seeds);
t_start = tic;

%% --- MAIN EXPERIMENTAL LOOP ---
for i = 1:length(N_hidden_range)
    n_hid = N_hidden_range(i);
    n_total = N_obs + n_hid;
    
    seeds_err_A = zeros(num_seeds, 1);
    seeds_err_W = zeros(num_seeds, 1);
    
    fprintf('Testing N_hidden = %d: ', n_hid);
    
    for s = 1:num_seeds
        current_seed = s * 1000 + n_hid; 
        
        % 1. Generate Ground Truth
        [A_full, ~, ~] = generate_graph(n_total, current_seed);
        A_full = max(0, A_full);
        A_full = A_full ./ max(A_full(:)); 
        
        obs_idx = 1:N_obs;
        hid_idx = N_obs+1:n_total;
        
        A_true_obs = A_full(obs_idx, obs_idx);
        W_true = A_full(obs_idx, hid_idx);
        
        % 2. Generate Dynamics
        t_vec = 0:del_t:T_duration;
        Nt = length(t_vec);
        X_obs_all = zeros(N_obs, Nt, N_proc);
        
        for pp = 1:N_proc
            X_full = generate_si_dynamics(A_full, t_vec, beta, current_seed + pp, division_factor);
            X_obs_all(:,:,pp) = X_full(obs_idx, :);
        end
        
        % 3. Inference
        [A_hat, W_hat, ~, ~] = PO_model2(X_obs_all, beta, del_t, n_hid, max_iter, tol, A_full);
        
        % 4. Store Errors
        seeds_err_A(s) = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
        
        % Handle W error only if hidden nodes exist
        if n_hid > 0
            seeds_err_W(s) = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
        else
            seeds_err_W(s) = NaN; 
        end
    end
    
    % Calculate Statistics
    mean_err_A(i) = mean(seeds_err_A);
    std_err_A(i)  = std(seeds_err_A);
    
    if n_hid > 0
        mean_err_W(i) = mean(seeds_err_W, 'omitnan');
        std_err_W(i)  = std(seeds_err_W, 'omitnan');
    else
        mean_err_W(i) = 0;
        std_err_W(i)  = 0;
    end
    
    fprintf('Done. Mean Err A: %.2f%%\n', mean_err_A(i)*100);
end

total_time = toc(t_start);
fprintf('\nExperiment Complete in %.2f minutes.\n', total_time/60);

%% --- VISUALIZATION ---
figure('Color', 'black', 'Position', [100, 100, 800, 500]);

% Plot A_OO error
errorbar(N_hidden_range, mean_err_A * 100, std_err_A * 100, '-o', 'Color', [0 0.447 0.741], ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'A_{OO} (Observed-Observed)');
hold on;

% Plot W error (starting from first hidden node)
idx_w = N_hidden_range > 0;
errorbar(N_hidden_range(idx_w), mean_err_W(idx_w) * 100, std_err_W(idx_w) * 100, '-s', 'Color', [0.85 0.325 0.098], ...
    'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'W (Observed-Hidden)');

grid on;
xlabel('Number of Hidden Nodes (N_{hidden})', 'FontSize', 12);
ylabel('Relative Recovery Error (%)', 'FontSize', 12);
title(sprintf('Network Recovery Error vs. Hidden Nodes (N_{obs}=%d)', N_obs), 'FontSize', 14);
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

% Annotate K=0 point
text(0.2, mean_err_A(1)*100, ' Fully Observed Case', 'FontSize', 10, 'FontWeight', 'bold');