%% --- MULTI-SEED EXPERIMENT: SI PARTIAL OBSERVABILITY ---
clear; clc; close all;

% Add paths for your local structure
addpath(fullfile('..', 'src'));

%% --- EXPERIMENT CONFIGURATION ---

num_seeds = 100;          % Number of random seeds to test
start_seed = 1;           % Starting seed value
run_id = 5;               % Run ID for organizing results

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

% Create results directory
results_dir = fullfile('..', 'results', sprintf('run_%d', run_id));
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

fprintf('=== MULTI-SEED SI EXPERIMENT ===\n');
fprintf('Configuration: N_obs=%d, N_hid=%d, K=%d, P=%d, Init=%s\n', ...
    N_obs, N_hidden, K, N_proc, init_mode);
fprintf('Running %d seeds (%d to %d)\n\n', num_seeds, start_seed, start_seed + num_seeds - 1);

%% --- INITIALIZE RESULTS STORAGE ---
results = struct();
results.seed = zeros(num_seeds, 1);
results.error_percentage = zeros(num_seeds, 1);
results.avg_degree = zeros(num_seeds, 1);
results.min_degree = zeros(num_seeds, 1);
results.max_degree = zeros(num_seeds, 1);

% Optional: store additional metrics
results.iterations = zeros(num_seeds, 1);
results.error_W = zeros(num_seeds, 1);

%% --- RUN EXPERIMENTS ---
fprintf('Progress:\n');
fprintf('%-6s | %-10s | %-10s | %-10s | %-10s | %-6s\n', ...
    'Seed', 'Error (%)', 'Avg Deg', 'Min Deg', 'Max Deg', 'Iters');
fprintf('%s\n', repmat('-', 1, 70));

for idx = 1:num_seeds
    seed = start_seed + idx - 1;
    
    try
        %% --- STEP 1: GENERATE GROUND TRUTH & DYNAMICS ---
        [A_full, ~, ~] = generate_graph(N_total, seed);
        A_full = max(0, A_full);
        A_full = A_full ./ max(A_full(:)); % Normalized Ground Truth
        
        % Calculate degree statistics for the full graph
        degrees = sum(A_full, 2);
        avg_degree = mean(degrees);
        min_degree = min(degrees);
        max_degree = max(degrees);
        
        % Extract partitions 
        obs_idx = 1:N_obs;
        A_true_obs = A_full(obs_idx, obs_idx);
        W_true = A_full(obs_idx, N_obs+1:end);
        
        % Generate Multiple SI Processes 
        t_vec = 0:del_t:T_duration;
        X_obs_all = zeros(N_obs, length(t_vec), N_proc);
        
        for pp = 1:N_proc
            X_full = generate_si_dynamics(A_full, t_vec, beta, seed + pp, division_factor);
            X_obs_all(:,:,pp) = X_full(obs_idx, :);
        end
        
        %% --- STEP 2: RUN INFERENCE ---
        [A_hat, W_hat, Z_hat, hist] = PO_model1( ...
            X_obs_all, beta, del_t, K, max_iter, tol, method, A_full, init_mode);
        
        %% --- STEP 3: ANALYZE RESULTS ---
        final_err_A = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
        final_err_W = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
        num_iterations = length(hist.obj);
        
        %% --- STEP 4: STORE RESULTS ---
        results.seed(idx) = seed;
        results.error_percentage(idx) = final_err_A * 100;
        results.avg_degree(idx) = avg_degree;
        results.min_degree(idx) = min_degree;
        results.max_degree(idx) = max_degree;
        results.iterations(idx) = num_iterations;
        results.error_W(idx) = final_err_W * 100;
        
        % Print progress
        fprintf('%-6d | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %-6d\n', ...
            seed, final_err_A * 100, avg_degree, min_degree, max_degree, num_iterations);
        
    catch ME
        fprintf('%-6d | ERROR: %s\n', seed, ME.message);
        % Store NaN for failed runs
        results.seed(idx) = seed;
        results.error_percentage(idx) = NaN;
        results.avg_degree(idx) = NaN;
        results.min_degree(idx) = NaN;
        results.max_degree(idx) = NaN;
        results.iterations(idx) = NaN;
        results.error_W(idx) = NaN;
    end
end

fprintf('%s\n', repmat('-', 1, 70));
fprintf('Completed %d experiments\n\n', num_seeds);

%% --- SAVE RESULTS TO CSV ---
% Create table for easy CSV export
results_table = table(...
    results.seed, ...
    results.error_percentage, ...
    results.avg_degree, ...
    results.min_degree, ...
    results.max_degree, ...
    results.iterations, ...
    results.error_W, ...
    'VariableNames', {'seed', 'error_percentage', 'avg_degree', ...
                      'min_degree', 'max_degree', 'iterations', 'error_W_percentage'});

% Save to CSV
csv_filename = fullfile(results_dir, 'seed_experiment_results.csv');
writetable(results_table, csv_filename);
fprintf('Results saved to: %s\n\n', csv_filename);

% Also save as .mat for further analysis
mat_filename = fullfile(results_dir, 'seed_experiment_results.mat');
save(mat_filename, 'results', 'results_table');

%% --- SUMMARY STATISTICS ---
fprintf('=== SUMMARY STATISTICS ===\n\n');

% Remove any NaN values for statistics
valid_idx = ~isnan(results.error_percentage);
num_valid = sum(valid_idx);

fprintf('Valid runs: %d / %d (%.1f%%)\n\n', num_valid, num_seeds, 100*num_valid/num_seeds);

if num_valid > 0
    fprintf('Error Percentage:\n');
    fprintf('  Mean:   %.2f%%\n', mean(results.error_percentage(valid_idx)));
    fprintf('  Median: %.2f%%\n', median(results.error_percentage(valid_idx)));
    fprintf('  Std:    %.2f%%\n', std(results.error_percentage(valid_idx)));
    fprintf('  Min:    %.2f%%\n', min(results.error_percentage(valid_idx)));
    fprintf('  Max:    %.2f%%\n', max(results.error_percentage(valid_idx)));
    fprintf('\n');
    
    fprintf('Average Degree:\n');
    fprintf('  Mean:   %.2f\n', mean(results.avg_degree(valid_idx)));
    fprintf('  Median: %.2f\n', median(results.avg_degree(valid_idx)));
    fprintf('  Std:    %.2f\n', std(results.avg_degree(valid_idx)));
    fprintf('  Min:    %.2f\n', min(results.avg_degree(valid_idx)));
    fprintf('  Max:    %.2f\n', max(results.avg_degree(valid_idx)));
    fprintf('\n');
    
    fprintf('Iterations:\n');
    fprintf('  Mean:   %.1f\n', mean(results.iterations(valid_idx)));
    fprintf('  Median: %.1f\n', median(results.iterations(valid_idx)));
    fprintf('  Min:    %d\n', min(results.iterations(valid_idx)));
    fprintf('  Max:    %d\n', max(results.iterations(valid_idx)));
end

%% --- OPTIONAL: CREATE SUMMARY PLOTS ---
if num_valid > 10  % Only create plots if we have enough data
    figure('Position', [100, 100, 1400, 900]);
    
    % Plot 1: Error vs Seed
    subplot(2, 3, 1);
    plot(results.seed(valid_idx), results.error_percentage(valid_idx), 'b.-');
    xlabel('Seed'); ylabel('Error (%)');
    title('Error vs Seed');
    grid on;
    
    % Plot 2: Error Distribution
    subplot(2, 3, 2);
    histogram(results.error_percentage(valid_idx), 20, 'FaceColor', 'b', 'EdgeColor', 'k');
    xlabel('Error (%)'); ylabel('Frequency');
    title('Error Distribution');
    grid on;
    
    % Plot 3: Error vs Average Degree
    subplot(2, 3, 3);
    scatter(results.avg_degree(valid_idx), results.error_percentage(valid_idx), 50, 'filled');
    xlabel('Average Degree'); ylabel('Error (%)');
    title('Error vs Average Degree');
    grid on;
    
    % Plot 4: Degree Distribution
    subplot(2, 3, 4);
    histogram(results.avg_degree(valid_idx), 20, 'FaceColor', 'g', 'EdgeColor', 'k');
    xlabel('Average Degree'); ylabel('Frequency');
    title('Average Degree Distribution');
    grid on;
    
    % Plot 5: Iterations vs Error
    subplot(2, 3, 5);
    scatter(results.iterations(valid_idx), results.error_percentage(valid_idx), 50, 'filled');
    xlabel('Iterations'); ylabel('Error (%)');
    title('Iterations vs Error');
    grid on;
    
    % Plot 6: Box plot comparison
    subplot(2, 3, 6);
    data_for_box = [results.error_percentage(valid_idx), results.error_W(valid_idx)];
    boxplot(data_for_box, 'Labels', {'A Error', 'W Error'});
    ylabel('Error (%)');
    title('Error Comparison');
    grid on;
    
    sgtitle(sprintf('Multi-Seed Experiment Summary (N=%d seeds)', num_valid));
    
    % Save figure
    fig_filename = fullfile(results_dir, 'seed_experiment_summary.png');
    saveas(gcf, fig_filename);
    fprintf('Summary figure saved to: %s\n', fig_filename);
end

fprintf('\n=== EXPERIMENT COMPLETE ===\n');