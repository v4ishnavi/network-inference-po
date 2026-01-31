% Plot_T_vs_Teff.m
clear; clc; close all;
addpath(fullfile('..', 'bin'));
addpath(fullfile('..', 'src'));

% --- SETUP ---
N_obs = 25;
N_total = 30;
T_max_samples = 100; % We will grow T from 2 to 400
beta = 0.5;
del_t = 0.05;
svd_tol = 5e-2; % Noise floor for rank
seed = 5;

% Generate 1 Long Process
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full/max(A_full(:)));
t_full = 0:del_t:(T_max_samples*del_t);
X_full = generate_si_dynamics(A_full, t_full, beta, seed, 1000);
X_obs = X_full(1:N_obs, :, 1); % Take observed nodes only

% --- ANALYSIS LOOP ---
T_values = 2:T_max_samples;
T_eff_values = zeros(size(T_values));

fprintf('Running Rank Analysis...\n');
for i = 1:length(T_values)
    current_T = T_values(i);
    
    % 1. Slice the data up to current T
    X_slice = X_obs(:, 1:current_T);
    
    % 2. Take Derivative (Focus on Dynamics)
    dX = diff(X_slice, 1, 2);
    
    % 3. Compute Numerical Rank (T_eff)
    s = svd(dX, 'econ');
    T_eff_values(i) = sum(s/max(s) > svd_tol);
    fprintf('T = %d: T_eff = %d\n', current_T, T_eff_values(i));
    % Print all singular values (eigenvalues of dX'*dX)
    fprintf('T = %d: Singular values: ', current_T);
    fprintf('%.6f ', s);
    fprintf('\n');
end

% --- PLOT ---
figure('Position', [100, 100, 700, 500]);
plot(T_values, T_eff_values, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
grid on;

% Add the "Ceiling" line
yline(N_obs, 'k--', 'Sensor Limit (N_{obs})', 'LineWidth', 2);

xlabel('Total Time Samples Stored (T)');
ylabel('Effective Information Rank (T_{eff})');
title(['Information Saturation: Why T_{eff} \leq N_{obs}']);
legend('T_{eff} (Data Rank)', 'Theoretical Max');

% Annotate the zones
text(5, 2, 'Linear Zone', 'FontSize', 12, 'Color', 'b');
text(T_max_samples/2, N_obs-0.5, 'Saturation / Ceiling', 'FontSize', 12, 'Color', 'b');