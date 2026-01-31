% Run partial-observability SI experiment
% Generates SI data with hidden nodes, saves the data,
% runs algorithm_PO_si with both random and LS initialization, and compares results.

clear; clc; close all;

% Add bin and src folders to path
addpath(fullfile('..', 'bin'));
addpath(fullfile('..', 'src'));

%% PARAMETERS
N_obs = 10;               % number of observed nodes
N_hidden = 2;            % number of hidden nodes
N_total = N_obs + N_hidden;
K = N_hidden;            % latent rank = number of hidden nodes
beta = 1.0;              % infection rate
del_t = 0.01;
T_end = 5.0;
t = 0:del_t:T_end;
N_process = 25;          % number of independent processes
division_factor = 1000;
seed = 42;
max_iter = 200;
tol = 1e-3;
method = "abs";          % residual processing method

fprintf('=== PARTIAL OBSERVABILITY SI EXPERIMENT ===\n');
fprintf('N_obs=%d, N_hidden=%d, K=%d, N_process=%d\n\n', N_obs, N_hidden, K, N_process);

%% Generate graph and SI dynamics
fprintf('Generating full graph (N_total=%d) and SI dynamics...\n', N_total);
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:));

X_full = zeros(N_total, length(t), N_process);
for pp = 1:N_process
    X_full(:,:,pp) = generate_si_dynamics(A_full, t, beta, 1000 + pp, division_factor);
end

% Extract observed data
obs_idx = 1:N_obs;
X_obs = X_full(obs_idx, :, :);
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, N_obs+1:end);

% Save generated data to data folder
data_dir = fullfile('..', 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
save_name = sprintf('Experiment_SI_partialobs_Nobs%d_Nhid%d_proc%d.mat', ...
                    N_obs, N_hidden, N_process);
save(fullfile(data_dir, save_name), 'X_full', 'X_obs', 't', 'del_t', ...
     'beta', 'A_full', 'obs_idx');
fprintf('Saved SI data to %s\n', fullfile(data_dir, save_name));

%% Method 1: Random initialization
fprintf('\n=== METHOD 1: Random Initialization ===\n');
tic;
[A_hat_rand, W_hat_rand, Z_hat_rand, hist_rand] = algorithm_PO_si( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'random');
time_rand = toc;

A_err_rand = 100 * norm(A_hat_rand - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
W_err_rand = 100 * norm(W_hat_rand - W_true, 'fro') / norm(W_true, 'fro');
fprintf('Random Init: A_err=%.2f%%, W_err=%.2f%%, Time=%.2fs, Iters=%d\n', ...
        A_err_rand, W_err_rand, time_rand, length(hist_rand.obj));

%% Method 2: Least Squares initialization
fprintf('\n=== METHOD 2: Least Squares Initialization ===\n');
tic;
[A_hat_ls, W_hat_ls, Z_hat_ls, hist_ls] = algorithm_PO_si( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'ls');
time_ls = toc;

A_err_ls = 100 * norm(A_hat_ls - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
W_err_ls = 100 * norm(W_hat_ls - W_true, 'fro') / norm(W_true, 'fro');
fprintf('LS Init: A_err=%.2f%%, W_err=%.2f%%, Time=%.2fs, Iters=%d\n', ...
        A_err_ls, W_err_ls, time_ls, length(hist_ls.obj));

%% Comparison Summary
fprintf('\n=== COMPARISON SUMMARY ===\n');
fprintf('Method    | A Error  | W Error  | Time   | Iters\n');
fprintf('----------|----------|----------|--------|----- \n');
fprintf('Random    | %7.2f%% | %7.2f%% | %5.2fs | %4d\n', ...
        A_err_rand, W_err_rand, time_rand, length(hist_rand.obj));
fprintf('LS        | %7.2f%% | %7.2f%% | %5.2fs | %4d\n', ...
        A_err_ls, W_err_ls, time_ls, length(hist_ls.obj));

if A_err_rand < A_err_ls
    fprintf('Winner: Random (%.2f%% better A error)\n', A_err_ls - A_err_rand);
else
    fprintf('Winner: LS (%.2f%% better A error)\n', A_err_rand - A_err_ls);
end

%% Visualization
plots_dir = fullfile('..', 'plots');
if ~exist(plots_dir, 'dir')
    mkdir(plots_dir);
end

figure('Position', [100, 100, 1600, 1000]);

% Row 1: Convergence plots
subplot(3,5,1);
semilogy(hist_rand.obj, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.obj, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective (log)');
legend('Random', 'LS', 'Location', 'best');
title('Objective Convergence'); grid on;

subplot(3,5,2);
semilogy(hist_rand.dA, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.dA, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('A Error (log)');
legend('Random', 'LS', 'Location', 'best');
title('A Error vs Truth'); grid on;

subplot(3,5,3);
semilogy(hist_rand.dW, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.dW, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('W Error (log)');
legend('Random', 'LS', 'Location', 'best');
title('W Error vs Truth'); grid on;

subplot(3,5,4);
semilogy(hist_rand.R_norm, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.R_norm, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Residual Norm (log)');
legend('Random', 'LS', 'Location', 'best');
title('Residual Magnitude'); grid on;

subplot(3,5,5);
bar([A_err_rand, A_err_ls; W_err_rand, W_err_ls]');
set(gca, 'XTickLabel', {'Random', 'LS'});
ylabel('Error (%)');
legend('A Error', 'W Error', 'Location', 'best');
title('Final Errors'); grid on;

% Row 2: A matrices
subplot(3,5,6);
imagesc(A_true_obs); colorbar; axis square;
title('True A_{OO}'); colormap('hot');

subplot(3,5,7);
imagesc(A_hat_rand); colorbar; axis square;
title(sprintf('Random A (%.1f%% err)', A_err_rand));

subplot(3,5,8);
imagesc(A_hat_ls); colorbar; axis square;
title(sprintf('LS A (%.1f%% err)', A_err_ls));

subplot(3,5,9);
imagesc(abs(A_hat_rand - A_true_obs)); colorbar; axis square;
title('|Random - True|');

subplot(3,5,10);
imagesc(abs(A_hat_ls - A_true_obs)); colorbar; axis square;
title('|LS - True|');

% Row 3: W matrices
subplot(3,5,11);
imagesc(W_true); colorbar; axis square;
title('True W');

subplot(3,5,12);
imagesc(W_hat_rand); colorbar; axis square;
title(sprintf('Random W (%.1f%% err)', W_err_rand));

subplot(3,5,13);
imagesc(W_hat_ls); colorbar; axis square;
title(sprintf('LS W (%.1f%% err)', W_err_ls));

subplot(3,5,14);
scatter(A_true_obs(:), A_hat_rand(:), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(A_true_obs(:), A_hat_ls(:), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
plot([0 max(A_true_obs(:))], [0 max(A_true_obs(:))], 'k--', 'LineWidth', 2);
xlabel('True A'); ylabel('Estimated A');
legend('Random', 'LS', 'y=x', 'Location', 'best');
title('A Scatter Plot'); grid on; axis square;

subplot(3,5,15);
scatter(W_true(:), W_hat_rand(:), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(W_true(:), W_hat_ls(:), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
max_W = max([W_true(:); W_hat_rand(:); W_hat_ls(:)]);
plot([0 max_W], [0 max_W], 'k--', 'LineWidth', 2);
xlabel('True W'); ylabel('Estimated W');
legend('Random', 'LS', 'y=x', 'Location', 'best');
title('W Scatter Plot'); grid on; axis square;

sgtitle(sprintf('Partial Observability SI: N_{obs}=%d, N_{hid}=%d, K=%d', ...
                N_obs, N_hidden, K));

% Save plot
plot_name = sprintf('partialobs_comparison_Nobs%d_Nhid%d.png', N_obs, N_hidden);
saveas(gcf, fullfile(plots_dir, plot_name));
saveas(gcf, fullfile(plots_dir, strrep(plot_name, '.png', '.fig')));
fprintf('\nPlot saved to %s\n', fullfile(plots_dir, plot_name));

%% Save results
results_dir = fullfile('..', 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
results_name = sprintf('partialobs_results_Nobs%d_Nhid%d_proc%d.mat', ...
                       N_obs, N_hidden, N_process);
save(fullfile(results_dir, results_name), ...
     'A_hat_rand', 'A_hat_ls', 'W_hat_rand', 'W_hat_ls', ...
     'A_true_obs', 'W_true', 'A_err_rand', 'A_err_ls', ...
     'W_err_rand', 'W_err_ls', 'time_rand', 'time_ls', ...
     'hist_rand', 'hist_ls', 'N_obs', 'N_hidden', 'K', 'save_name');
fprintf('Results saved to %s\n', fullfile(results_dir, results_name));

fprintf('\nDone. To run this experiment from the repository root, run:\n');
fprintf('>> experiments/run_partial_observability_si\n');
