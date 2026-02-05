% Plot Kuramoto oscillator dynamics
% Generates a random graph and simulates Kuramoto model
% dθ_i/dt = ω_i + K * Σ_j A_ij * sin(θ_j - θ_i)

clear; close all; clc;

% Add path to graph generation function
addpath('../src');
addpath('../bin');

%% Hyperparameters
K = 3.6;              % Coupling strength 
% this gives us the <r> ~ 0.5 regime for N=10.
dt_sim = 0.1;         % Time step for RK4 integration
N = 10;               % Number of oscillators
T_total = 100;        % Total simulation time
graph_seed = 42;      % Seed for graph generation
data_seed = 123;      % Seed for initial conditions

%% Generate graph (same as SI dynamics)
% Generate adjacency matrix with Gaussian weights scaled to [0,1]
[A, ~, ~] = generate_graph(N, graph_seed);
A = max(0, A);                    % Ensure non-negative
A = A ./ max(A(:));               % Scale to [0,1]

fprintf('Generated graph with N=%d nodes\n', N);
fprintf('Max coupling weight: %.3f\n', max(A(:)));
fprintf('Mean coupling weight: %.3f\n', mean(A(:)));

%% Generate time vector
t = 0:dt_sim:T_total;

%% Simulate Kuramoto dynamics
fprintf('Simulating Kuramoto dynamics with K=%.2f, dt=%.2f...\n', K, dt_sim);
Theta = generate_kuramoto_dynamics(A, t, K, data_seed, dt_sim);

%% Plotting
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Phase trajectories over time
subplot(2, 2, 1);
plot(t, Theta', 'LineWidth', 1.5);
xlabel('Time', 'FontSize', 12);
ylabel('Phase θ_i (radians)', 'FontSize', 12);
title('Phase Trajectories of All Oscillators', 'FontSize', 14);
grid on;
xlim([0, T_total]);
ylim([0, 2*pi]);
set(gca, 'YTick', [0, pi/2, pi, 3*pi/2, 2*pi], ...
         'YTickLabel', {'0', 'π/2', 'π', '3π/2', '2π'});

% Plot 2: Order parameter over time
subplot(2, 2, 2);
r = compute_order_parameter(Theta);
plot(t, r, 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
xlabel('Time', 'FontSize', 12);
ylabel('Order Parameter r(t)', 'FontSize', 12);
title('Synchronization Order Parameter', 'FontSize', 14);
grid on;
xlim([0, T_total]);
ylim([0, 1]);

% Plot 3: Phase space (first 3 oscillators)
subplot(2, 2, 3);
if N >= 3
    plot3(cos(Theta(1,:)), sin(Theta(1,:)), t, 'LineWidth', 1.5); hold on;
    plot3(cos(Theta(2,:)), sin(Theta(2,:)), t, 'LineWidth', 1.5);
    plot3(cos(Theta(3,:)), sin(Theta(3,:)), t, 'LineWidth', 1.5);
    xlabel('cos(θ)', 'FontSize', 12);
    ylabel('sin(θ)', 'FontSize', 12);
    zlabel('Time', 'FontSize', 12);
    title('Phase Space (First 3 Oscillators)', 'FontSize', 14);
    legend('Osc 1', 'Osc 2', 'Osc 3', 'Location', 'best');
    grid on; view(45, 30);
end

% Plot 4: Snapshot of phases on unit circle
subplot(2, 2, 4);
t_snapshots = [10, 30, 50, T_total];
colors = lines(4);
for idx = 1:length(t_snapshots)
    t_snap = t_snapshots(idx);
    [~, tidx] = min(abs(t - t_snap));
    theta_snap = Theta(:, tidx);
    
    % Plot on unit circle
    polarscatter(theta_snap, ones(N,1), 50, colors(idx,:), 'filled'); hold on;
end
title('Phase Distribution at Different Times', 'FontSize', 14);
legend(arrayfun(@(x) sprintf('t=%.0f', x), t_snapshots, 'UniformOutput', false), ...
       'Location', 'best');
set(gca, 'ThetaZeroLocation', 'top');

sgtitle(sprintf('Kuramoto Oscillators (N=%d, K=%.2f, dt=%.2f)', N, K, dt_sim), ...
        'FontSize', 16, 'FontWeight', 'bold');

%% Additional analysis
fprintf('\n=== Simulation Results ===\n');
fprintf('Final order parameter: r(T) = %.4f\n', r(end));
fprintf('Mean order parameter: <r> = %.4f\n', mean(r));
fprintf('System %s\n', ternary(mean(r) > 0.5, 'synchronized', 'incoherent'));

%% Helper functions

function r = compute_order_parameter(Theta)
    % Compute Kuramoto order parameter: r = |1/N * Σ exp(i*θ_j)|
    N = size(Theta, 1);
    complex_phases = exp(1i * Theta);
    mean_phase = mean(complex_phases, 1);
    r = abs(mean_phase);
end

function result = ternary(condition, true_val, false_val)
    % Simple ternary operator
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
