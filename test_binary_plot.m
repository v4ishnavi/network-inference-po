
clear;

%% random seeds
graph_seed = 10;
data_seed = 1001;

binary_graph = 1;
add_noise = 0;

%% graph generation
N = 15;
if binary_graph == 0
    [A, D, L] = generate_graph(N,graph_seed);
else
    [A, D, L] = generate_graph_binary(N,graph_seed);
end

%% data generation
T = 500;
del_t = 0.001;
t = 0:del_t:T*del_t;

N_process = 2;
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
Lhat_MT = graph_learning_MT(X,del_t);

Lhat_LS = graph_learning_proposed_LS(X,del_t);

Lhat_LS2 = graph_learning_proposed_LS2(X,del_t);

if binary_graph == 1
    Lhat_MT = round(Lhat_MT);
    Lhat_LS = round(Lhat_LS);
    Lhat_LS2 = round(Lhat_LS2);
end

%% error
Lerror_MT = 100*norm(Lhat_MT-L,"fro")/norm(L,"fro");
disp(Lerror_MT);

Lerror_LS = 100*norm(Lhat_LS-L,"fro")/norm(L,"fro");
disp(Lerror_LS);

Lerror_LS2 = 100*norm(Lhat_LS2-L,"fro")/norm(L,"fro");
disp(Lerror_LS2);


%% data plotting

figure;
subplot(221); imagesc(L); colorbar;
subplot(224); imagesc(Lhat_MT); colorbar;
subplot(222); imagesc(Lhat_LS); colorbar;
subplot(223); imagesc(Lhat_LS2); colorbar;
