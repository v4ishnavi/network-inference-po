function Theta = generate_kuramoto_dynamics(A, t, K, data_seed, dt_sim)
% Generate Kuramoto dynamics data using RK4 integration
% Theta = generate_kuramoto_dynamics(A, t, K, data_seed, dt_sim)
% 
% Inputs:
%   A: adjacency matrix (N x N) - coupling weights between oscillators
%   t: time vector for sampling
%   K: coupling strength
%   data_seed: random seed for initial conditions and natural frequencies
%   dt_sim: time step for RK4 integration (default: 0.1)
%
% Output:
%   Theta: phase angles (N x length(t))
%
% Dynamics:
%   dθ_i/dt = ω_i + K * Σ_j A_ij * sin(θ_j - θ_i)

    if nargin < 5
        dt_sim = 0.1;  % Default time step
    end

    N = size(A, 1);
    Nt = length(t);

    rng(data_seed);
    
    % Natural frequencies ~ N(0,1)
    omega = randn(N, 1);
    
    % Initial phases ~ U[0, 2π] (fully disordered state)
    theta0 = 2 * pi * rand(N, 1);

    % Create fine-grained simulation time vector
    t_sim = t(1):dt_sim:t(end);
    
    % Initialize phase trajectory
    theta_sim = zeros(N, length(t_sim));
    theta_sim(:, 1) = theta0;

    % RK4 integration
    for i = 2:length(t_sim)
        theta_curr = theta_sim(:, i-1);
        
        % RK4 stages
        k1 = kuramoto_derivative(theta_curr, omega, A, K);
        k2 = kuramoto_derivative(theta_curr + dt_sim/2 * k1, omega, A, K, N);
        k3 = kuramoto_derivative(theta_curr + dt_sim/2 * k2, omega, A, K, N);
        k4 = kuramoto_derivative(theta_curr + dt_sim * k3, omega, A, K, N);
        
        % Update: θ(t+Δt) = θ(t) + Δt/6 * (k1 + 2*k2 + 2*k3 + k4)
        theta_sim(:, i) = theta_curr + dt_sim/6 * (k1 + 2*k2 + 2*k3 + k4);
        
        % Apply modulo 2π to keep phases in [0, 2π]
        theta_sim(:, i) = mod(theta_sim(:, i), 2*pi);
    end

    % Sample at requested time points
    sample_indices = round((t - t(1)) / dt_sim) + 1;
    sample_indices = min(sample_indices, length(t_sim));  % Ensure within bounds
    Theta = theta_sim(:, sample_indices);
end


function dtheta_dt = kuramoto_derivative(theta, omega, A, K, N)
% Compute time derivative for Kuramoto model
% dθ_i/dt = ω_i + K * Σ_j A_ij * sin(θ_j - θ_i)
    
    N = length(theta);
    dtheta_dt = omega;  % Start with natural frequencies
    
    % Add coupling term
    for i = 1:N
        coupling_sum = 0;
        for j = 1:N
            coupling_sum = coupling_sum + A(i,j) * sin(theta(j) - theta(i));
        end
        dtheta_dt(i) = dtheta_dt(i) + (K/N) * coupling_sum;
    end 
end