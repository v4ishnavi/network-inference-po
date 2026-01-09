% Vary number of observed nodes: N_hidden=1, N_obs=[2,3,4,5,6,7,8], K=1
% Compare random vs LS initialization across different observed node counts

clear; clc; close all;

%% PARAMETERS
N_hidden = 1;
N_obs_values = [4, 5, 6, 7, 8];
N_process = 20;
K = N_hidden;  % K=1 always
beta = 1.0;
del_t = 0.01;
T_end = 5.0;
t = 0:del_t:T_end;
max_iter = 200;
tol = 1e-3;
method = "abs";

% Storage for results
results = struct();
results.N_obs_values = N_obs_values;
results.A_err_rand = zeros(size(N_obs_values));
results.A_err_ls = zeros(size(N_obs_values));
results.W_err_rand = zeros(size(N_obs_values));
results.W_err_ls = zeros(size(N_obs_values));
results.time_rand = zeros(size(N_obs_values));
results.time_ls = zeros(size(N_obs_values));
results.iters_rand = zeros(size(N_obs_values));
results.iters_ls = zeros(size(N_obs_values));

fprintf('=== VARYING OBSERVED NODES EXPERIMENT ===\n');
fprintf('N_hidden=%d, K=%d, N_process=%d\n\n', N_hidden, K, N_process);

division_factor = 10;

%% Loop over different numbers of observed nodes
for idx = 1:length(N_obs_values)
    N_obs = N_obs_values(idx);
    N_total = N_obs + N_hidden;
    
    fprintf('--- Testing N_obs=%d (total=%d) ---\n', N_obs, N_total);
    
    % Generate data
    [A_full, ~, ~] = generate_graph(N_total, 42 + idx);  % Different seed for each
    A_full = max(0, A_full);
    A_full = A_full ./ max(A_full(:));
    
    X_full = zeros(N_total, length(t), N_process);
    for pp = 1:N_process
        X_full(:,:,pp) = generate_si_dynamics(A_full, t, beta, 1000 + pp + idx*100, division_factor);
    end
    
    obs_idx = 1:N_obs;
    X_obs = X_full(obs_idx,:,:);
    A_true_obs = A_full(obs_idx, obs_idx);
    W_true = A_full(obs_idx, N_obs+1:end);
    
    % Random initialization
    fprintf('  Random init... ');
    tic;
    [A_hat_rand, W_hat_rand, ~, hist_rand] = algorithm_PO_si( ...
        X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'random');
    results.time_rand(idx) = toc;
    results.A_err_rand(idx) = 100 * norm(A_hat_rand - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
    results.W_err_rand(idx) = 100 * norm(W_hat_rand - W_true, 'fro') / norm(W_true, 'fro');
    results.iters_rand(idx) = length(hist_rand.obj);
    fprintf('A_err=%.2f%%, time=%.2fs\n', results.A_err_rand(idx), results.time_rand(idx));
    
    % Least squares initialization
    fprintf('  LS init... ');
    tic;
    [A_hat_ls, W_hat_ls, ~, hist_ls] = algorithm_PO_si( ...
        X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'ls');
    results.time_ls(idx) = toc;
    results.A_err_ls(idx) = 100 * norm(A_hat_ls - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
    results.W_err_ls(idx) = 100 * norm(W_hat_ls - W_true, 'fro') / norm(W_true, 'fro');
    results.iters_ls(idx) = length(hist_ls.obj);
    fprintf('A_err=%.2f%%, time=%.2fs\n', results.A_err_ls(idx), results.time_ls(idx));
    
    fprintf('\n');
end

%% Summary table
fprintf('=== RESULTS SUMMARY ===\n');
fprintf('N_obs | Random A_err | LS A_err | Random W_err | LS W_err | Random Time | LS Time\n');
fprintf('------|--------------|----------|--------------|----------|-------------|--------\n');
for idx = 1:length(N_obs_values)
    fprintf('%5d | %11.2f%% | %7.2f%% | %11.2f%% | %7.2f%% | %10.2fs | %6.2fs\n', ...
        N_obs_values(idx), results.A_err_rand(idx), results.A_err_ls(idx), ...
        results.W_err_rand(idx), results.W_err_ls(idx), ...
        results.time_rand(idx), results.time_ls(idx));
end

%% Plot results
figure('Position', [100, 100, 1200, 600]);

subplot(2,3,1);
plot(N_obs_values, results.A_err_rand, 'ro-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(N_obs_values, results.A_err_ls, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Observed Nodes'); ylabel('A Error (%)');
legend('Random', 'LS', 'Location', 'best');
title('A Reconstruction Error'); grid on;

subplot(2,3,2);
plot(N_obs_values, results.W_err_rand, 'ro-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(N_obs_values, results.W_err_ls, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Observed Nodes'); ylabel('W Error (%)');
legend('Random', 'LS', 'Location', 'best');
title('W Reconstruction Error'); grid on;

subplot(2,3,3);
plot(N_obs_values, results.time_rand, 'ro-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(N_obs_values, results.time_ls, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Observed Nodes'); ylabel('Time (seconds)');
legend('Random', 'LS', 'Location', 'best');
title('Computation Time'); grid on;

subplot(2,3,4);
plot(N_obs_values, results.iters_rand, 'ro-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(N_obs_values, results.iters_ls, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Observed Nodes'); ylabel('Iterations');
legend('Random', 'LS', 'Location', 'best');
title('Convergence Iterations'); grid on;

subplot(2,3,5);
improvement = results.A_err_ls - results.A_err_rand;
bar(N_obs_values, improvement);
xlabel('Number of Observed Nodes'); ylabel('A Error Difference (LS - Random)');
title('Random vs LS (positive = Random better)'); grid on;
yline(0, 'k--', 'LineWidth', 1);

subplot(2,3,6);
semilogy(N_obs_values, results.A_err_rand, 'ro-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogy(N_obs_values, results.A_err_ls, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Observed Nodes'); ylabel('A Error (%) - Log Scale');
legend('Random', 'LS', 'Location', 'best');
title('A Error (Log Scale)'); grid on;

%% Analyze scaling behavior
fprintf('\n=== SCALING ANALYSIS ===\n');
% Fit power law to errors vs N_obs
log_N = log(N_obs_values);
log_err_rand = log(results.A_err_rand);
log_err_ls = log(results.A_err_ls);

% Remove any -Inf values (zero errors)
valid_rand = isfinite(log_err_rand);
valid_ls = isfinite(log_err_ls);

if sum(valid_rand) > 1
    p_rand = polyfit(log_N(valid_rand), log_err_rand(valid_rand), 1);
    fprintf('Random scaling: A_error ∝ N_obs^%.2f\n', p_rand(1));
end

if sum(valid_ls) > 1
    p_ls = polyfit(log_N(valid_ls), log_err_ls(valid_ls), 1);
    fprintf('LS scaling: A_error ∝ N_obs^%.2f\n', p_ls(1));
end

%% Save results
save('vary_observed_nodes_results.mat', 'results');
saveas(gcf, 'plots/vary_observed_nodes.png');
saveas(gcf, 'plots/vary_observed_nodes.fig');
fprintf('\nResults saved to vary_observed_nodes_results.mat\n');
