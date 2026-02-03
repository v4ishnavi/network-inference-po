%% --- SINGLE DEBUGGING EXPERIMENT: SI PARTIAL OBSERVABILITY (POD) ---
clear; clc; close all;

% Add paths
addpath(fullfile('..', 'src'));

%% --- GLOBAL CONFIGURATION ---
N_obs = 10;               
N_hidden = 2;             
N_total = N_obs + N_hidden;
K = N_hidden; % Rank matches hidden nodes for this debug run

beta = 0.5;               
del_t = 0.05;             
T_duration = 5.0;         
max_iter = 100;           
tol = 1e-6;               
N_proc = 10;
seed = 250;              

fprintf('=== SI POD RECONSTRUCTION RUN ===\n');

%% --- STEP 1: GENERATE GROUND TRUTH & DYNAMICS ---
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:)); 

obs_idx = 1:N_obs;
hid_idx = N_obs+1:N_total;
A_true_obs = A_full(obs_idx, obs_idx);
W_true = A_full(obs_idx, hid_idx);

t_vec = 0:del_t:T_duration;
Nt = length(t_vec);
X_obs_all = zeros(N_obs, Nt, N_proc);
X_hid_true_all = zeros(N_hidden, Nt, N_proc); 

fprintf('Generating %d SI processes...\n', N_proc);
for pp = 1:N_proc
    % Assuming generate_si_dynamics is your solver
    X_full = generate_si_dynamics(A_full, t_vec, beta, seed + pp, 1000);
    X_obs_all(:,:,pp) = X_full(obs_idx, :);
    X_hid_true_all(:,:,pp) = X_full(hid_idx, :); 
end

%% --- STEP 2: RUN INFERENCE (POD VERSION) ---
fprintf('Starting POD Alternating Minimization...\n');

% CALLING THE NEW POD MODEL
[A_hat, W_hat, C_hat, hist] = PO_model4( ...
    X_obs_all, beta, del_t, K, max_iter, tol, A_full);

%% --- STEP 3: ANALYZE RESULTS ---
final_err_A = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
final_err_W = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');

fprintf('\n--- Final Metrics ---\n');
fprintf('A_OO Relative Error: %.2f%%\n', final_err_A * 100);
fprintf('W Relative Error   : %.2f%%\n', final_err_W * 100);

%% --- STEP 4: VISUALIZATION ---
figure('Position', [50, 50, 1400, 800]);

% 1. Error Trajectories
subplot(2, 3, 1);
plot(hist.dA * 100, 'r-o', 'MarkerSize', 4, 'DisplayName', 'A Error'); hold on;
plot(hist.dW * 100, 'b-s', 'MarkerSize', 4, 'DisplayName', 'W Error');
grid on; xlabel('Iteration'); ylabel('Error (%)'); 
title('Reconstruction Convergence'); legend;

% 2. Objective Function
subplot(2, 3, 4);
semilogy(hist.obj, 'k', 'LineWidth', 1.5, 'Color', 'blue');
grid on; xlabel('Iteration'); ylabel('Residual MSE');
title('Objective Function');

% 3. Adjacency Comparison
subplot(2, 3, 2);
imagesc(A_true_obs); colorbar; axis square; title('True A_{OO}');
subplot(2, 3, 5);
imagesc(A_hat); colorbar; axis square; 
title(sprintf('Recovered A_{OO}\nErr: %.1f%%', final_err_A*100));

% 4. Latent Dynamics Comparison (The POD Logic)
proc_to_plot = [1, 2]; 
colors_true = [0, 1, 0]; % Black for truth
colors_hat = [1, 0, 0];  % Red for estimate

for i = 1:length(proc_to_plot)
    pp = proc_to_plot(i);
    subplot(2, 3, i * 3); 
    
    % --- KEY CHANGE FOR POD ---
    % Calculate Z using the learned mapping: Z = C * X_obs
    Z_est = C_hat * X_obs_all(:,:,pp); 
    
    % Plot Ground Truth (Hidden Node 1)
    plot(t_vec, X_hid_true_all(1, :, pp), 'Color', colors_true, 'LineWidth', 2.5, 'DisplayName', 'True Hidden'); 
    hold on;
    
    % Plot POD Estimate
    plot(t_vec, Z_est(1, :), '--', 'Color', colors_hat, 'LineWidth', 2, 'DisplayName', 'POD Estimate (C*x_o)');
    
    grid on; xlabel('Time'); ylabel('Infection Prob');
    title(sprintf('Hidden State Recovery (Proc %d)', pp));
    if i == 1, legend('Location', 'best'); end
end

sgtitle(sprintf('SI Topology Inference (POD Formulation): N_{obs}=%d, N_{hid}=%d', N_obs, N_hidden));