% Run full-observability SI experiment - Compare LS methods
% Generates SI data for a fully observed network, saves the data,
% runs both graph_learning_si_LS and graph_learning_si_LS2, and compares results.

clear; clc; close all;

% Add src folders to path for access to functions\
addpath(fullfile('..', 'src'));

%% PARAMETERS
N = 7;                   % total nodes (fully observed)
beta = 1.0;              % infection rate used to generate data
del_t = 0.01;
T_end = 5.0;
t = 0:del_t:T_end;
N_process = 1;          % number of independent processes/trajectories
division_factor = 10;    % passed to generate_si_dynamics
seed = 42;

%% Generate graph and SI dynamics
fprintf('Generating graph (N=%d) and SI dynamics (%d processes)...\n', N, N_process);
[A_full, ~, ~] = generate_graph(N, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:));

X_full = zeros(N, length(t), N_process);
for pp = 1:N_process
    X_full(:,:,pp) = generate_si_dynamics(A_full, t, beta, 1000 + pp, division_factor);
end

% Save generated data to data folder
data_dir = fullfile('..', 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
save_name = sprintf('Experiment_SI_fullobs_N%d_proc%d.mat', N, N_process);
save(fullfile(data_dir, save_name), 'X_full', 't', 'del_t', 'beta', 'A_full');
fprintf('Saved SI data to %s\n', fullfile(data_dir, save_name));

%% Method 1: graph_learning_si_LS (standard LS)
fprintf('\n=== METHOD 1: graph_learning_si_LS ===\n');
tic;
[A_hat_LS, beta_hat_LS] = graph_learning_si_LS(X_full, del_t, beta);
time_LS = toc;
fprintf('Finished in %.2f s (beta_hat=%.3f)\n', time_LS, beta_hat_LS);

%% Method 2: graph_learning_si_LS2 (symmetric half-vec LS)
fprintf('\n=== METHOD 2: graph_learning_si_LS2 ===\n');
tic;
[A_hat_LS2, beta_hat_LS2] = graph_learning_si_LS2(X_full, del_t, beta);
time_LS2 = toc;
fprintf('Finished in %.2f s (beta_hat=%.3f)\n', time_LS2, beta_hat_LS2);

%% Compare both estimates with ground truth
A_true = A_full; % full observability
if norm(A_true, 'fro') > 0
    A_err_LS = 100 * norm(A_hat_LS - A_true, 'fro') / norm(A_true, 'fro');
    A_err_LS2 = 100 * norm(A_hat_LS2 - A_true, 'fro') / norm(A_true, 'fro');
else
    A_err_LS = NaN;
    A_err_LS2 = NaN;
end

fprintf('\n=== COMPARISON SUMMARY ===\n');
fprintf('Method              | A Error  | Time   | Symmetric?\n');
fprintf('--------------------|----------|--------|-----------\n');
if isequal(A_hat_LS, A_hat_LS')
    sym_LS = 'Yes';
else
    sym_LS = 'No';
end
if isequal(A_hat_LS2, A_hat_LS2')
    sym_LS2 = 'Yes';
else
    sym_LS2 = 'No';
end
fprintf('graph_learning_si_LS  | %6.2f%% | %5.2fs | %s\n', A_err_LS, time_LS, sym_LS);
fprintf('graph_learning_si_LS2 | %6.2f%% | %5.2fs | %s\n', A_err_LS2, time_LS2, sym_LS2);

if A_err_LS < A_err_LS2
    fprintf('Winner: LS (%.2f%% better)\n', A_err_LS2 - A_err_LS);
else
    fprintf('Winner: LS2 (%.2f%% better)\n', A_err_LS - A_err_LS2);
end

% Compute difference between methods
method_diff = norm(A_hat_LS - A_hat_LS2, 'fro');
fprintf('Difference between methods: %.4f (Frobenius norm)\n', method_diff);

%% Visualization
plots_dir = fullfile('..', 'plots');
if ~exist(plots_dir, 'dir')
    mkdir(plots_dir);
end

figure('Position', [100,100,1400,900]);

% Row 1: True, LS, LS2
subplot(3,4,1);
imagesc(A_true); colorbar; axis square; 
title('True A'); colormap('hot');

subplot(3,4,2);
imagesc(A_hat_LS); colorbar; axis square;
title(sprintf('LS Method (%.2f%% err)', A_err_LS));

subplot(3,4,3);
imagesc(A_hat_LS2); colorbar; axis square;
title(sprintf('LS2 Method (%.2f%% err)', A_err_LS2));

subplot(3,4,4);
imagesc(abs(A_hat_LS - A_hat_LS2)); colorbar; axis square;
title('|LS - LS2| Difference');

% Row 2: Errors vs ground truth
subplot(3,4,5);
imagesc(abs(A_hat_LS - A_true)); colorbar; axis square;
title('|LS - True|');

subplot(3,4,6);
imagesc(abs(A_hat_LS2 - A_true)); colorbar; axis square;
title('|LS2 - True|');

subplot(3,4,7);
scatter(A_true(:), A_hat_LS(:), 20, 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot([0 max(A_true(:))], [0 max(A_true(:))], 'r--', 'LineWidth', 2);
xlabel('True A'); ylabel('Estimated A (LS)');
title('LS Scatter'); grid on; axis square;

subplot(3,4,8);
scatter(A_true(:), A_hat_LS2(:), 20, 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot([0 max(A_true(:))], [0 max(A_true(:))], 'r--', 'LineWidth', 2);
xlabel('True A'); ylabel('Estimated A (LS2)');
title('LS2 Scatter'); grid on; axis square;

% Row 3: Symmetry analysis
subplot(3,4,9);
imagesc(A_hat_LS - A_hat_LS'); colorbar; axis square;
title('LS Asymmetry (A - A^T)');
A_LS_T = A_hat_LS';
max_asym_LS = max(abs(A_hat_LS(:) - A_LS_T(:)));
if max_asym_LS > 0
    caxis([-1 1]*max_asym_LS);
end

subplot(3,4,10);
imagesc(A_hat_LS2 - A_hat_LS2'); colorbar; axis square;
title('LS2 Asymmetry (A - A^T)');
A_LS2_T = A_hat_LS2';
max_asym_LS2 = max(abs(A_hat_LS2(:) - A_LS2_T(:)));
if max_asym_LS2 > 0
    caxis([-1 1]*max_asym_LS2);
end

subplot(3,4,11);
bar([A_err_LS, A_err_LS2]);
set(gca, 'XTickLabel', {'LS', 'LS2'});
ylabel('Error (%)');
title('Reconstruction Error Comparison'); grid on;

subplot(3,4,12);
bar([time_LS, time_LS2]);
set(gca, 'XTickLabel', {'LS', 'LS2'});
ylabel('Time (s)');
title('Computation Time'); grid on;

sgtitle(sprintf('Full Observability SI Comparison (N=%d, %d processes)', N, N_process));
saveas(gcf, fullfile(plots_dir, sprintf('fullobs_LS_comparison_N%d.png', N)));
saveas(gcf, fullfile(plots_dir, sprintf('fullobs_LS_comparison_N%d.fig', N)));

%% Save results
results_dir = fullfile('..', 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
results_name = sprintf('fullobs_comparison_N%d_proc%d.mat', N, N_process);
save(fullfile(results_dir, results_name), 'A_hat_LS', 'A_hat_LS2', 'A_true', ...
     'A_err_LS', 'A_err_LS2', 'beta_hat_LS', 'beta_hat_LS2', ...
     'time_LS', 'time_LS2', 'save_name');
fprintf('\nResults saved to %s\n', fullfile(results_dir, results_name));
fprintf('Plot saved to %s\n', fullfile(plots_dir, sprintf('fullobs_LS_comparison_N%d.png', N)));

fprintf('\nDone. To run this experiment from the repository root, run:\n');
fprintf('>> experiments/run_full_observability_si\n');
