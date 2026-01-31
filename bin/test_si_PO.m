
clear; clc; close all;

%% PARAMETERS
N_obs      = 5;
N_hidden   = 2;
N_total    = N_obs + N_hidden;
N_process  = 10;        % Number of independent SI processes
K          = 2;        % Latent rank
beta       = 1.0;
del_t      = 0.01;
T_end      = 5.0;
t          = 0:del_t:T_end;

max_iter   = 200;  % Reduced from 500 - should converge faster
tol        = 1e-3;  % Relaxed from 1e-3
method = "abs";
%% Generate true full graph
[A_full, ~, ~] = generate_graph(N_total, 1);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:));

%% Simulate SI dynamics N_process times
division_factor = 10;
X_full = zeros(N_total, length(t), N_process);

for pp = 1:N_process
    X_full(:,:,pp) = generate_si_dynamics(A_full, t, beta, ...
                                          1000 + pp, division_factor);
end

%% Extract observed block for all processes
obs_idx = 1:N_obs;
X_obs = X_full(obs_idx,:,:);   % (N_obs x T x N_process)

%% Run EM learner for partial observability w/ multiple processes
init_mode = 'ls';  % Options: 'true', 'random', 'ls'
[A_hat, W_hat, Z_hat, hist] = algorithm_PO_si( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full, init_mode);

disp("DEBUG: is hist empty? " + isempty(hist.obj));

%% Compare with true observed adjacency
A_true_obs = A_full(obs_idx, obs_idx);

A_err = 100 * norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
fprintf('\n\n===== RESULTS =====\n');
fprintf('Frobenius error in A (%%): %.2f\n', A_err);

disp('Estimated A_{OO}:');
disp(A_hat);

disp('True A_{OO}:');
disp(A_true_obs);
fprintf('\n');
W_true = A_full(obs_idx, N_obs+1:end);
disp(size(W_true));
disp(size(W_hat));
W_err = 100 * norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');
fprintf('frobenius error in W (%%): %.2f\n', W_err);
disp('Estimated W (N_obs x K):');
disp(W_hat);
disp('True W (N_obs x K):');
W_true = A_full(obs_idx, N_obs+1:end);
disp(W_true);

%% Create detailed log file
if ~exist('plots', 'dir')
    mkdir('plots');
end

% Generate timestamp for unique log file
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
log_filename = sprintf('plots/experiment_log_%s_%s.txt', init_mode, timestamp);

% Open log file for writing
log_file = fopen(log_filename, 'w');

fprintf(log_file, '========================================\n');
fprintf(log_file, 'SI PARTIAL OBSERVABILITY EXPERIMENT LOG\n');
fprintf(log_file, '========================================\n');
fprintf(log_file, 'Timestamp: %s\n', timestamp);
fprintf(log_file, 'Initialization Mode: %s\n', init_mode);
fprintf(log_file, '\n--- EXPERIMENT PARAMETERS ---\n');
fprintf(log_file, 'N_obs: %d\n', N_obs);
fprintf(log_file, 'N_hidden: %d\n', N_hidden);
fprintf(log_file, 'N_total: %d\n', N_total);
fprintf(log_file, 'N_process: %d\n', N_process);
fprintf(log_file, 'K (latent rank): %d\n', K);
fprintf(log_file, 'beta: %.3f\n', beta);
fprintf(log_file, 'del_t: %.4f\n', del_t);
fprintf(log_file, 'T_end: %.1f\n', T_end);
fprintf(log_file, 'max_iter: %d\n', max_iter);
fprintf(log_file, 'tolerance: %.1e\n', tol);
fprintf(log_file, 'method: %s\n', method);

fprintf(log_file, '\n--- CONVERGENCE ANALYSIS ---\n');
fprintf(log_file, 'Total iterations completed: %d\n', length(hist.obj));
fprintf(log_file, 'Final objective value: %.6e\n', hist.obj(end));
fprintf(log_file, 'Initial objective value: %.6e\n', hist.obj(1));
fprintf(log_file, 'Objective reduction: %.6e\n', hist.obj(1) - hist.obj(end));

if length(hist.dA) > 0
    fprintf(log_file, '\n--- PARAMETER CHANGE TRACKING ---\n');
    fprintf(log_file, 'Final dA (A parameter change): %.6e\n', hist.dA(end));
    fprintf(log_file, 'Final dW (W parameter change): %.6e\n', hist.dW(end));
    fprintf(log_file, 'Final dZ (Z parameter change): %.6e\n', hist.dZ(end));
    fprintf(log_file, 'Average dA over iterations: %.6e\n', mean(hist.dA));
    fprintf(log_file, 'Average dW over iterations: %.6e\n', mean(hist.dW));
    fprintf(log_file, 'Average dZ over iterations: %.6e\n', mean(hist.dZ));
    fprintf(log_file, 'Std dA over iterations: %.6e\n', std(hist.dA));
    fprintf(log_file, 'Std dW over iterations: %.6e\n', std(hist.dW));
    fprintf(log_file, 'Std dZ over iterations: %.6e\n', std(hist.dZ));
end

fprintf(log_file, '\n--- RECONSTRUCTION ERRORS ---\n');
fprintf(log_file, 'A_oo Frobenius Error (%%): %.4f\n', A_err);
fprintf(log_file, 'W Frobenius Error (%%): %.4f\n', W_err);
fprintf(log_file, 'A_oo Absolute Error: %.6f\n', norm(A_hat - A_true_obs, 'fro'));
fprintf(log_file, 'W Absolute Error: %.6f\n', norm(W_hat - W_true, 'fro'));
fprintf(log_file, 'A_oo True Norm: %.6f\n', norm(A_true_obs, 'fro'));
fprintf(log_file, 'W True Norm: %.6f\n', norm(W_true, 'fro'));

fprintf(log_file, '\n--- MATRIX DETAILS ---\n');
fprintf(log_file, 'True A_oo matrix:\n');
for i = 1:size(A_true_obs,1)
    for j = 1:size(A_true_obs,2)
        fprintf(log_file, '%8.4f ', A_true_obs(i,j));
    end
    fprintf(log_file, '\n');
end

fprintf(log_file, '\nEstimated A_oo matrix:\n');
for i = 1:size(A_hat,1)
    for j = 1:size(A_hat,2)
        fprintf(log_file, '%8.4f ', A_hat(i,j));
    end
    fprintf(log_file, '\n');
end

fprintf(log_file, '\nTrue W matrix:\n');
for i = 1:size(W_true,1)
    for j = 1:size(W_true,2)
        fprintf(log_file, '%8.4f ', W_true(i,j));
    end
    fprintf(log_file, '\n');
end

fprintf(log_file, '\nEstimated W matrix:\n');
for i = 1:size(W_hat,1)
    for j = 1:size(W_hat,2)
        fprintf(log_file, '%8.4f ', W_hat(i,j));
    end
    fprintf(log_file, '\n');
end

fprintf(log_file, '\n--- Z ANALYSIS (AVERAGED ACROSS PROCESSES) ---\n');
fprintf(log_file, 'Z is estimated per time point and process\n');
fprintf(log_file, 'Z dimensions: %d x %d (K x total_time_points)\n', size(Z_hat,1), size(Z_hat,2));
fprintf(log_file, 'Z mean across all time points: %.6f\n', mean(Z_hat(:)));
fprintf(log_file, 'Z std across all time points: %.6f\n', std(Z_hat(:)));
fprintf(log_file, 'Z min value: %.6f\n', min(Z_hat(:)));
fprintf(log_file, 'Z max value: %.6f\n', max(Z_hat(:)));

% Z changes are tracked as changes between iterations (not across processes)
% The Z matrix represents hidden states for each time point across all processes
if length(hist.dZ) > 0
    fprintf(log_file, 'Z change tracking shows parameter updates between EM iterations\n');
    fprintf(log_file, 'This measures ||Z_new - Z_old||_F / ||Z_old||_F between iterations\n');
end

fprintf(log_file, '\n--- CONVERGENCE DETAILS ---\n');
if length(hist.obj) > 1
    obj_slope = (hist.obj(end) - hist.obj(max(1,end-5))) / min(5, length(hist.obj)-1);
    fprintf(log_file, 'Objective slope (last 5 iters): %.6e\n', obj_slope);
end

if length(hist.dA) > 1
    dA_trend = hist.dA(end) / hist.dA(1);
    dW_trend = hist.dW(end) / hist.dW(1);
    dZ_trend = hist.dZ(end) / hist.dZ(1);
    fprintf(log_file, 'dA final/initial ratio: %.6f\n', dA_trend);
    fprintf(log_file, 'dW final/initial ratio: %.6f\n', dW_trend);
    fprintf(log_file, 'dZ final/initial ratio: %.6f\n', dZ_trend);
end

fclose(log_file);
fprintf('Detailed log saved to %s\n', log_filename);



%% Plot Convergence Metrics
figure('Position', [100, 100, 1200, 800]);

subplot(2,3,1);
semilogy(hist.obj, 'b.-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('Objective ||Y - Phi θ||^2 (log)');
grid on;
title(sprintf('Objective Convergence (%s init)', init_mode));

subplot(2,3,2);
semilogy(hist.dA, 'r.-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('||A_{new} - A_{true}||_F / ||A_{true}||_F');
grid on;
title('A Error vs Ground Truth');

subplot(2,3,3);
semilogy(hist.dW, 'g.-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('||W_{new} - W_{true}||_F / ||W_{true}||_F');
grid on;
title('W Error vs Ground Truth');

subplot(2,3,4);
semilogy(hist.dZ, 'm.-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('||Z_{new} - Z_{prev}||_F / ||Z_{prev}||_F');
grid on;
title('Z Parameter Change (Between Iterations)');

subplot(2,3,5);
semilogy(hist.R_norm, 'c.-', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('||R||_F (Residual Norm)');
grid on;
title('Residual Magnitude');

subplot(2,3,6);
% Plot both A and W changes together
semilogy(hist.dA, 'r.-', 'LineWidth', 2, 'MarkerSize', 4); hold on;
semilogy(hist.dW, 'g.-', 'LineWidth', 2, 'MarkerSize', 4);
semilogy(hist.dZ, 'm.-', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Iteration');
ylabel('Relative Change');
legend('dA', 'dW', 'dZ', 'Location', 'best');
grid on;
title('All Parameter Changes');

% Save convergence plot with descriptive filename
conv_filename = sprintf('plots/convergence_%s_init_%s', init_mode, timestamp);
saveas(gcf, [conv_filename '.png']);
saveas(gcf, [conv_filename '.fig']);
fprintf('Convergence plot saved to %s.png\n', conv_filename);

%% Plot adjacency comparison
figure('Position', [100, 400, 800, 400]);
subplot(1,2,1); imagesc(A_true_obs); colorbar; axis square;
title('True A_{OO}');
subplot(1,2,2); imagesc(A_hat); colorbar; axis square;
title(sprintf('Estimated A_{OO} (%s init)', init_mode));

% Save adjacency plot with descriptive filename
adj_filename = sprintf('plots/adjacency_%s_init_%s', init_mode, timestamp);
saveas(gcf, [adj_filename '.png']);
saveas(gcf, [adj_filename '.fig']);
fprintf('Adjacency plot saved to %s.png\n', adj_filename);
