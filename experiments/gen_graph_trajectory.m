clear; clc; close all;

% Add paths if your functions are in subfolders
addpath(fullfile('..', 'bin'));
addpath(fullfile('..', 'src'));

%% 1. PARAMETERS
N_obs = 25;                % Observed nodes
N_hidden = 5;              % Hidden nodes
N_total = N_obs + N_hidden;
beta = 0.5;                % Infection rate
del_t = 0.05;
T_end = 5.0;
t = 0:del_t:T_end;
seed = 5;                 % Fixed seed for reproducibility

%% 2. GENERATE SYSTEM
fprintf('Generating graph and dynamics for 1 process...\n');

% Generate the Graph Structure (Adjacency Matrix)
[A_full, ~, ~] = generate_graph(N_total, seed);
A_full = max(0, A_full);
A_full = A_full ./ max(A_full(:)); % Normalize

% Generate Dynamics for ONE Process
% We use a specific seed (e.g., 1001) so we get a specific infection starting point
start_node_seed = 1001; 
division_factor = 1000;
X_single = generate_si_dynamics(A_full, t, beta, start_node_seed, division_factor);

%% 3. VISUALIZATION
figure('Position', [100, 100, 1200, 500]);

% --- Subplot 1: Time Series Trajectories ---
subplot(1, 2, 1);
hold on;

% Plot Observed Nodes (Blue lines)
h_obs = plot(t, X_single(1:N_obs, :), 'b-', 'LineWidth', 1.5);

% Plot Hidden Nodes (Red dashed lines)
h_hid = plot(t, X_single(N_obs+1:end, :), 'r--', 'LineWidth', 2);

% Formatting
xlabel('Time (t)');
ylabel('Infection Probability x_i(t)');
title('SI Dynamics Trajectories');
grid on;
ylim([0, 1.05]);

% Create custom legend (since we have multiple lines per group)
legend([h_obs(1), h_hid(1)], ...
       {sprintf('Observed (%d)', N_obs), sprintf('Hidden (%d)', N_hidden)}, ...
       'Location', 'southeast');

% --- Subplot 2: Adjacency Matrix (The Structure) ---
subplot(1, 2, 2);
imagesc(A_full);
colorbar;
colormap('hot');
axis square;
title('Underlying Graph Adjacency Matrix');
xlabel('Node Index');
ylabel('Node Index');

% Draw box around Observed block
rectangle('Position', [0.5, 0.5, N_obs, N_obs], 'EdgeColor', 'b', 'LineWidth', 2);
text(N_obs/2, 0, 'Observed', 'Color', 'b', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

% Draw box around Hidden interaction
rectangle('Position', [N_obs+0.5, 0.5, N_hidden, N_total], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
text(N_total, N_total/2, 'Hidden Links', 'Color', 'r', 'Rotation', -90, 'VerticalAlignment', 'bottom');

fprintf('Plot generated.\n');