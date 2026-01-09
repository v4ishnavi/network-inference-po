clear;

%% random seeds
graph_seed = 10;
data_seed = 1001;
binary_graph = 1;  % test with binary graphs
add_noise = 0;

%% graph generation
N = 8;
if binary_graph == 0
    [~, ~, L] = generate_graph(N, graph_seed);
else
    [~, ~, L] = generate_graph_binary(N, graph_seed);
end

% Convert to adjacency for SI
A = -(L - diag(diag(L)));
A = max(0, A);
A = 0.5 * (A + A');

%% SI parameters
beta = 2.0;
T_end = 1.5;
T = 500;
del_t = 0.005;

% t = 0:del_t:T_end;
t = 0:del_t:del_t*T;
division_factor = 10; 
N_process = 2;
X = [];
for ii = 1:N_process
    new_data = generate_si_dynamics(A, t, beta, data_seed+ii, division_factor);
    X(:,:,ii) = new_data;
end

if add_noise == 1
    noise_sigma = 0.01;
    X = X + noise_sigma*randn(size(X));
    X = max(0, min(1, X));
end

%% graph learning
[Ahat_MT, beta_hat_MT] = graph_learning_si_MT(X, del_t, beta);

[Ahat_LS, beta_hat_LS] = graph_learning_si_LS(X, del_t, beta);

[Ahat_LS2, beta_hat_LS2] = graph_learning_si_LS2(X, del_t, beta);

if binary_graph == 1
    Ahat_MT = round(Ahat_MT);
    Ahat_LS = round(Ahat_LS);
    Ahat_LS2 = round(Ahat_LS2);
    disp(Ahat_LS2)
    disp(Ahat_LS)
    disp(Ahat_MT)
end

%% error
Aerror_MT = 100*norm(Ahat_MT-A,"fro")/norm(A,"fro");
disp(['MT Error: ', num2str(Aerror_MT, '%.1f'), '%']);

Aerror_LS = 100*norm(Ahat_LS-A,"fro")/norm(A,"fro");
disp(['LS Error: ', num2str(Aerror_LS, '%.1f'), '%']);

Aerror_LS2 = 100*norm(Ahat_LS2-A,"fro")/norm(A,"fro");
disp(['LS2 Error: ', num2str(Aerror_LS2, '%.1f'), '%']);

%% data plotting
figure('Position', [100, 100, 800, 600]);
subplot(221); 
imagesc(A); colorbar; title('True A'); axis square;

subplot(222); 
imagesc(Ahat_MT); colorbar; title(['MT (', num2str(Aerror_MT, '%.1f'), '%)']); axis square;

subplot(223); 
imagesc(Ahat_LS); colorbar; title(['LS (', num2str(Aerror_LS, '%.1f'), '%)']); axis square;

subplot(224); 
imagesc(Ahat_LS2); colorbar; title(['LS2 (', num2str(Aerror_LS2, '%.1f'), '%)']); axis square;

sgtitle('SI Dynamics - Binary Graph Learning');
