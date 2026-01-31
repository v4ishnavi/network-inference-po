
clear;
required_files = {
    'generate_graph.m',
    'generate_graph_binary.m', 
    'generate_heat_diffusion.m',
    'graph_learning_MT.m',
    'graph_learning_proposed_LS.m',
    'graph_learning_proposed_LS2.m'
};

for i = 1:length(required_files)
    if exist(required_files{i}, 'file') == 2
        fprintf('✓ %s exists\n', required_files{i});
    else
        fprintf('✗ %s MISSING\n', required_files{i});
    end
end
%% random seeds
graph_seed = 1;
data_seed = 1001;

binary_graph = 0;
add_noise = 0;

%% graph generation
N = 5;
if binary_graph == 0
    [A, D, L] = generate_graph(N,graph_seed);
else
    [A, D, L] = generate_graph_binary(N,graph_seed);
end
% disp(A); disp(D); disp(L);

%% data generation
T = 1000;
del_t = 0.001;
t = 0:del_t:T*del_t;
N_process = 3;
X = [];
for ii = 1:N_process
    new_data = generate_heat_diffusion(L,t,data_seed+ii);
    X(:,:,ii) = new_data;
end

if add_noise == 1
    % add noise
    noise_sigma = 0.0001;
    X = X + noise_sigma*randn(size(X));
end

%% graph learning
disp("Graph learning::: MT");
Lhat_MT = graph_learning_MT(X,del_t);

disp("Graph learning::: LS");
Lhat_LS = graph_learning_proposed_LS(X,del_t);

disp("Graph learning::: LS2");
Lhat_LS2 = graph_learning_proposed_LS2(X,del_t);

%% error
Lerror_MT = 100*norm(Lhat_MT-L,"fro")/norm(L,"fro");
disp("The error for MT is:");
disp(Lerror_MT);

Lerror_LS = 100*norm(Lhat_LS-L,"fro")/norm(L,"fro");
disp("The error for LS is:");
disp(Lerror_LS);

Lerror_LS2 = 100*norm(Lhat_LS2-L,"fro")/norm(L,"fro");
disp("The error for LS2 is:");
disp(Lerror_LS2);

%% data plotting
figure; 
for ii = 1:N_process
    subplot(1,N_process,ii);
    plot(X(:,:,ii)'); grid on;
end

figure;
subplot(221); imagesc(L); colorbar;
subplot(224); imagesc(Lhat_MT); colorbar;
subplot(222); imagesc(Lhat_LS); colorbar;
subplot(223); imagesc(Lhat_LS2); colorbar;

% figure;
% subplot(221); imagesc(L); colorbar;
% subplot(224); imagesc(Lhat_MT-L); colorbar;
% subplot(222); imagesc(Lhat_LS-L); colorbar;
% subplot(223); imagesc(Lhat_LS2-L); colorbar;
