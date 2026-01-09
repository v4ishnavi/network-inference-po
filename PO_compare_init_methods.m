% Compare random vs least squares initialization for SI partial observability
% Runs both methods on the same data and compares results

clear; clc; close all;

%% PARAMETERS
N_obs      = 5;
N_hidden   = 2;
N_total    = N_obs + N_hidden;
N_process  = 20;        
K          = N_hidden;  % Latent rank = number of hidden nodes
beta       = 1.0;
del_t      = 0.01;
T_end      = 5.0;
t          = 0:del_t:T_end;

max_iter   = 200;
tol        = 1e-3;
method     = "abs";

fprintf('=== INITIALIZATION COMPARISON ===\n');
fprintf('N_obs=%d, N_hidden=%d, K=%d, N_process=%d\n', N_obs, N_hidden, K, N_process);

%% Generate true full graph and data
[A_full, ~, ~] = generate_graph(N_total, 42);  % Fixed seed for reproducibility
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:));

division_factor = 10;
X_full = zeros(N_total, length(t), N_process);

fprintf('Generating SI dynamics...\n');
for pp = 1:N_process
    X_full(:,:,pp) = generate_si_dynamics(A_full, t, beta, 1000 + pp, division_factor);
end

obs_idx = 1:N_obs;
X_obs = X_full(obs_idx,:,:);
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, N_obs+1:end);

%% Method 1: Random initialization
fprintf('\n--- RANDOM INITIALIZATION ---\n');
tic;
[A_hat_rand, W_hat_rand, Z_hat_rand, hist_rand] = algorithm_PO_si( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'random');
time_rand = toc;

A_err_rand = 100 * norm(A_hat_rand - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
W_err_rand = 100 * norm(W_hat_rand - W_true, 'fro') / norm(W_true, 'fro');

fprintf('Random Init Results:\n');
fprintf('  A error: %.2f%%, W error: %.2f%%\n', A_err_rand, W_err_rand);
fprintf('  Time: %.2f sec, Iterations: %d\n', time_rand, length(hist_rand.obj));

%% Method 2: Least Squares initialization
fprintf('\n--- LEAST SQUARES INITIALIZATION ---\n');
tic;
[A_hat_ls, W_hat_ls, Z_hat_ls, hist_ls] = algorithm_PO_si( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full, 'ls');
time_ls = toc;

A_err_ls = 100 * norm(A_hat_ls - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
W_err_ls = 100 * norm(W_hat_ls - W_true, 'fro') / norm(W_true, 'fro');

fprintf('LS Init Results:\n');
fprintf('  A error: %.2f%%, W error: %.2f%%\n', A_err_ls, W_err_ls);
fprintf('  Time: %.2f sec, Iterations: %d\n', time_ls, length(hist_ls.obj));

%% Summary comparison
fprintf('\n=== SUMMARY ===\n');
fprintf('Method       | A Error | W Error | Time  | Iters\n');
fprintf('-------------|---------|---------|-------|----- \n');
fprintf('Random       | %6.2f%% | %6.2f%% | %5.2fs | %4d\n', A_err_rand, W_err_rand, time_rand, length(hist_rand.obj));
fprintf('Least Squares| %6.2f%% | %6.2f%% | %5.2fs | %4d\n', A_err_ls, W_err_ls, time_ls, length(hist_ls.obj));

if A_err_rand < A_err_ls
    fprintf('Winner: Random initialization (%.2f%% better A error)\n', A_err_ls - A_err_rand);
else
    fprintf('Winner: Least Squares initialization (%.2f%% better A error)\n', A_err_rand - A_err_ls);
end

%% Plot comparison
figure('Position', [100, 100, 1400, 800]);

% Convergence comparison
subplot(2,4,1);
semilogy(hist_rand.obj, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.obj, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective (log)');
legend('Random', 'LS', 'Location', 'best');
title('Objective Convergence'); grid on;

subplot(2,4,2);
semilogy(hist_rand.dA, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.dA, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('A Error (log)');
legend('Random', 'LS', 'Location', 'best');
title('A Error vs Truth'); grid on;

subplot(2,4,3);
semilogy(hist_rand.dW, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.dW, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('W Error (log)');
legend('Random', 'LS', 'Location', 'best');
title('W Error vs Truth'); grid on;

subplot(2,4,4);
semilogy(hist_rand.R_norm, 'r.-', 'LineWidth', 2); hold on;
semilogy(hist_ls.R_norm, 'b.-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Residual Norm (log)');
legend('Random', 'LS', 'Location', 'best');
title('Residual Magnitude'); grid on;

% Matrix comparisons
subplot(2,4,5);
imagesc(A_true_obs); colorbar; axis square;
title('True A_{OO}'); 

subplot(2,4,6);
imagesc(A_hat_rand); colorbar; axis square;
title(sprintf('Random A (%.1f%% err)', A_err_rand));

subplot(2,4,7);
imagesc(A_hat_ls); colorbar; axis square;
title(sprintf('LS A (%.1f%% err)', A_err_ls));

subplot(2,4,8);
imagesc(abs(A_hat_rand - A_hat_ls)); colorbar; axis square;
title('|Random - LS| Difference');

%% Save results
save('init_comparison_results.mat', 'A_err_rand', 'A_err_ls', 'W_err_rand', 'W_err_ls', ...
     'time_rand', 'time_ls', 'hist_rand', 'hist_ls', 'N_obs', 'N_hidden', 'K');
% save plot to file
if ~exist('plots', 'dir')
    mkdir('plots');
end
saveas(gcf, fullfile('plots', 'init_comparison.png'));
saveas(gcf, fullfile('plots', 'init_comparison.fig'));

fprintf('Plot saved to plots/init_comparison.png\n');
fprintf('\nResults saved to init_comparison_results.mat\n');
