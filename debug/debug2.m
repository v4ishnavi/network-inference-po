%% --- SINGLE DEBUGGING EXPERIMENT: SI PARTIAL OBSERVABILITY ---
clear; clc; close all;

% Add paths for your local structure
addpath(fullfile('..', 'src'));

%% --- GLOBAL CONFIGURATION ---
run_id = 4;
N_obs = 12;               
N_hidden = 3;             
N_total = N_obs + N_hidden;
K = N_hidden;             

beta = 0.5;               
del_t = 0.05;             
T_duration = 5.0;         
division_factor = 1000;   

max_iter = 500;           
tol = 1e-5;               
N_proc = 8;               
seed = 500;              

fprintf('=== SI DEBUGGdebugING RUN ===\n');
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
X_hid_true_all = zeros(N_hidden, Nt, N_proc); % <--- Added to store ground truth

fprintf('Generating %d SI processes...\n', N_proc);
for pp = 1:N_proc
    X_full = generate_si_dynamics(A_full, t_vec, beta, seed + pp, division_factor);
    X_obs_all(:,:,pp) = X_full(obs_idx, :);
    X_hid_true_all(:,:,pp) = X_full(hid_idx, :); % <--- Store True Hidden
end
%% --- STEP 2: RUN INFERENCE ---
fprintf('Starting PO_model2 iterations...\n');
% Ensure PO_model2 returns Z_hat
[A_hat, W_hat, Z_hat, hist] = PO_model2( ...
    X_obs_all, beta, del_t, K, max_iter, tol, A_full);

%% --- STEP 3: ANALYZE RESULTS ---
final_err_A = norm(A_hat - A_true_obs, 'fro') / norm(A_true_obs, 'fro');
final_err_W = norm(W_hat - W_true, 'fro') / norm(W_true, 'fro');

fprintf('\n--- Final Metrics ---\n');
fprintf('A_OO Relative Error: %.2f%%\n', final_err_A * 100);
fprintf('W Relative Error   : %.2f%%\n', final_err_W * 100);

%% --- STEP 4: VISUALIZATION ---
figure('Position', [50, 50, 1500, 800]); % Wider figure for 3 columns

% 1. Parameter Recovery Trajectory (Top Left)
subplot(2, 3, 1);
plot(hist.dA * 100, 'r-s', 'LineWidth', 1.2, 'DisplayName', 'A Error'); hold on;
plot(hist.dW * 100, 'g-^', 'LineWidth', 1.2, 'DisplayName', 'W Error');
grid on; xlabel('Iteration'); ylabel('Error (%)'); 
title('A & W Recovery Trajectory'); legend;

% 2. Objective Function (Bottom Left)
subplot(2, 3, 4);
semilogy(hist.obj, 'b', 'LineWidth', 1.5);
grid on; xlabel('Iteration'); ylabel('MSE');
title('Objective Function Descent');

% 3. Adjacency Comparison (Middle Column)
subplot(2, 3, 2);
imagesc(A_true_obs); colorbar; axis square;
title('True A_{OO}');

subplot(2, 3, 5);
imagesc(A_hat); colorbar; axis square;
title(sprintf('Recovered A_{OO}\n(Err: %.1f%%)', final_err_A*100));

% 4. Latent Dynamics Comparison (Right Column)
proc_to_plot = [1, 2]; % Choosing first two processes
colors_true = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980]; 
colors_hat = [0.3010, 0.7450, 0.9330; 0.9290, 0.6940, 0.1250];

for i = 1:length(proc_to_plot)
    pp = proc_to_plot(i);
    subplot(2, 3, i * 3); % This targets index 3 and 6 (Right Column)
    
    % Slice Z_hat for current process
    start_idx = (pp-1)*(Nt-1) + 1;
    end_idx = pp*(Nt-1);
    Z_proc = Z_hat(:, start_idx:end_idx);
    t_mid = t_vec(1:end-1) + del_t/2; 
    
    % Plot Ground Truth Hidden Nodes
    h1 = plot(t_vec, X_hid_true_all(1, :, pp), 'Color', colors_true(1,:), 'LineWidth', 2); hold on;
    h2 = plot(t_vec, X_hid_true_all(2, :, pp), 'Color', colors_true(2,:), 'LineWidth', 2);
    
    % Plot Estimated Latent States (Dashed)
    h3 = plot(t_mid, Z_proc(1, :), '--', 'Color', colors_hat(1,:), 'LineWidth', 2);
    h4 = plot(t_mid, Z_proc(2, :), '--', 'Color', colors_hat(2,:), 'LineWidth', 2);
    
    grid on; xlabel('Time'); ylabel('Infection Prob');
    title(sprintf('Latent States (Process %d)', pp));
    
    if i == 1
        legend([h1, h2, h3, h4], {'True z_1','True z_2','Est z_1','Est z_2'}, ...
            'Location', 'best', 'FontSize', 8);
    end
end

sgtitle(sprintf('SI Partial Observability: N_{obs}=%d, N_{hid}=%d, P=%d', N_obs, N_hidden, N_proc));