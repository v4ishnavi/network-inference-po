clear;

%% random seeds
graph_seed = 1;
data_seed = 1001;

binary_graph = 0;
add_noise = 0;

%% graph generation
N = 10;  % smaller for SI testing
if binary_graph == 0
    [A, ~, ~] = generate_graph(N, graph_seed);
else
    [A, ~, ~] = generate_graph_binary(N, graph_seed);
end

disp("[DEBUG] Generated graph with Laplacian L:");
% Convert Laplacian to adjacency for SI dynamics
% Better conversion
% D = diag(diag(L));
% A = D - L;
disp(A)
A = max(0, A);
A = A ./ max(A(:));
disp(A)

%% SI parameters
beta = 1.0;  % infection rate
T_end = 5.0;  % total time
del_t = 0.01;  % sampling time step
t = 0:del_t:T_end;
division_factor = 10; 
N_process = 2;
X = [];
for ii = 1:N_process
    new_data = generate_si_dynamics(A, t, beta, data_seed+ii, division_factor);
    X(:,:,ii) = new_data;
end
% disp("[DEBUG] Generated SI dynamics data X with size:");
% disp(size(X));
if add_noise == 1
    % add noise
    noise_sigma = 0.01;
    X = X + noise_sigma*randn(size(X));
    if any(X(:) > 1 | X(:) < 0)
    fprintf("Values out of bounds detected in X after noise addition.\n");
    fprintf("Max value: %.4f, Min value: %.4f\n", max(X(:)), min(X(:)));
    end
    X = max(0, min(1, X));
end
% disp("[DEBUG] SI dynamics data X after noise (if added):");
% disp(size(X));
disp(size(X))  % shape: (N, length(t), N_process)
%disp(X(:,:,1))  % Display the first process data f/
%% graph learning
[Ahat_MT, beta_hat_MT] = graph_learning_si_MT(X, del_t, beta);

[Ahat_LS, beta_hat_LS] = graph_learning_si_LS(X, del_t, beta);

[Ahat_LS2, beta_hat_LS2] = graph_learning_si_LS2(X, del_t, beta);

if binary_graph == 1
    Ahat_MT = round(Ahat_MT);
    Ahat_LS = round(Ahat_LS);
    Ahat_LS2 = round(Ahat_LS2);
end

%% error computation
Aerror_MT = 100*norm(Ahat_MT-A,"fro")/norm(A,"fro");
disp(['MT Error: ', num2str(Aerror_MT, '%.2f'), '%']);

Aerror_LS = 100*norm(Ahat_LS-A,"fro")/norm(A,"fro");
disp(['LS Error: ', num2str(Aerror_LS, '%.2f'), '%']);

Aerror_LS2 = 100*norm(Ahat_LS2-A,"fro")/norm(A,"fro");
disp(['LS2 Error: ', num2str(Aerror_LS2, '%.2f'), '%']);

%% beta estimation errors: here we just set beta_hat = beta for simplicity
disp(['True beta: ', num2str(beta, '%.2f')]);
disp(['MT beta: ', num2str(beta_hat_MT, '%.2f')]);
disp(['LS beta: ', num2str(beta_hat_LS, '%.2f')]);
disp(['LS2 beta: ', num2str(beta_hat_LS2, '%.2f')]);

%% data plotting
figure('Position', [100, 100, 1200, 400]); 
for ii = 1:N_process
    subplot(1, N_process, ii);
    plot(t, X(:,:,ii)'); 
    grid on;
    xlabel('Time');
    ylabel('Infection Probability');
    title(['Process ', num2str(ii)]);
end

%% adjacency matrix comparison
figure('Position', [100, 500, 1200, 300]);
subplot(1,4,1); imagesc(A); colorbar; title('True A'); axis square;
subplot(1,4,2); imagesc(Ahat_MT); colorbar; title('MT'); axis square;
subplot(1,4,3); imagesc(Ahat_LS); colorbar; title('LS'); axis square;
subplot(1,4,4); imagesc(Ahat_LS2); colorbar; title('LS2'); axis square;
