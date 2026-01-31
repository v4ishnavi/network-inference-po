function X = generate_si_dynamics(A, t, beta, data_seed, division_factor)
% Generate SI dynamics data
% X = generate_si_dynamics(A, t, beta, data_seed)
% 
% Inputs:
%   A: adjacency matrix (N x N)
%   t: time vector for sampling
%   beta: infection rate
%   data_seed: random seed for initial conditions
%
% Output:
%   X: infection probabilities (N x length(t))

N = size(A, 1);
Nt = length(t);
% disp("[DEBUG] N (number of nodes):");
% disp(N);
rng(data_seed);
x0 = 0.05 * rand(N, 1);

% Use smaller time step for simulation
dt_sim = min(diff(t)) / division_factor; 
t_sim = t(1):dt_sim:t(end);
% disp("[DEBUG] t_sim size:");
% disp(size(t_sim));
% disp("[DEBUG] dt_sim:");
% disp(dt_sim);
% Simulate with Euler method
x_sim = zeros(N, length(t_sim));
x_sim(:, 1) = x0;
% disp("[DEBUG] x_sim initial state:");
% disp(x_sim(:, 1));
% disp("[DEBUG] checking for NaNs in initial state:");
% disp(sum(isnan(x_sim(:, 1))));
for i = 2:length(t_sim)
    x_curr = x_sim(:, i-1);
    dx_dt = beta * (1 - x_curr) .* (A * x_curr);
    x_sim(:, i) = x_curr + dt_sim * dx_dt;
    % if sum(isnan(x_sim(:, i))) > 0
    %     disp("[DEBUG] NaNs found in x_sim at step " + num2str(i) + ":");
    %     disp(sum(isnan(x_sim(:, i))));
    % end
    x_sim(:, i) = max(0, min(1, x_sim(:, i)));
    % if sum(isnan(x_sim(:, i))) > 0
    %     disp("[DEBUG] NaNs found in x_sim at step " + num2str(i) + ":");
    %     disp(sum(isnan(x_sim(:, i))));
    % end
end
% Sample every division_factor points instead of interpolating
sample_indices = 1:division_factor:length(t_sim);
sample_indices = sample_indices(1:Nt);  % Ensure we get exactly Nt points
% disp("[DEBUG] sample_indices:");
% disp(sample_indices);
X = x_sim(:, sample_indices);
% disp("[DEBUG] X size after sampling:");
% disp(size(X));
% disp("[DEBUG] checking for NaNs in X:");
% disp(sum(isnan(X(:))));

end


% disp("[DEBUG] Finished simulation of SI dynamics.============================");
% X = zeros(N, Nt);
% fprintf('[DEBUG] t_sim: [%f, %f], length=%d\n', t_sim(1), t_sim(end), length(t_sim));
% fprintf('[DEBUG] t: [%f, %f], length=%d\n', t(1), t(end), length(t));
% fprintf('[DEBUG] t points outside t_sim range: %d\n', sum(t < t_sim(1) | t > t_sim(end)));
% for i = 1:N
%     % we interpolate because t_sim may not align with t
%     X(i, :) = interp1(t_sim, x_sim(i, :), t, 'linear');
%     % disp("[DEBUG] Interpolated node " + num2str(i) + " data:");
%     % disp(X(i, :));
%     % disp("[DEBUG] checking for NaNs in interpolated data:");
%     % disp(sum(isnan(X(i, :))));
% end
% disp("[DEBUG] Finished interpolation of SI dynamics data.===================");
% end
