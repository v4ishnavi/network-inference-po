
clear;

binary_graph = 0;
add_noise = 1;

% data sampling
T = 1000;
del_t = 0.001;
t = 0:del_t:T*del_t;

Num_graphs = 10;
graph_node_num_set = 5:15;
N_process = 1;

error_MT_All = zeros(numel(graph_node_num_set),Num_graphs);
error_LS_All = zeros(numel(graph_node_num_set),Num_graphs);
error_LS2_All = zeros(numel(graph_node_num_set),Num_graphs);

parpool(4);

for iN = 1:numel(graph_node_num_set)

    N = graph_node_num_set(iN);
    disp(iN);

    %% graph generation
    parfor iGraph = 1:Num_graphs
        graph_seed = iN*100 + iGraph;
        if binary_graph == 0
            [A, D, L] = generate_graph(N,graph_seed);
        else
            [A, D, L] = generate_graph_binary(N,graph_seed);
        end

        %% data generation
        data_seed = iN*10000 + iGraph;
        
        X = [];
        for ii = 1:N_process
            new_data = generate_heat_diffusion(L,t,data_seed+ii);
            X(:,:,ii) = new_data;
        end

        % add noise
        if add_noise == 1
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
        
        Lerror_MT = 100*norm(Lhat_MT-L,"fro")/norm(L,"fro");
        error_MT_All(iN,iGraph) = Lerror_MT;

        Lerror_LS = 100*norm(Lhat_LS-L,"fro")/norm(L,"fro");
        error_LS_All(iN,iGraph) = Lerror_LS;

        Lerror_LS2 = 100*norm(Lhat_LS2-L,"fro")/norm(L,"fro");
        error_LS2_All(iN,iGraph) = Lerror_LS2;

    end
end

delete(gcp('nocreate'));

figure; 
plot(graph_node_num_set,mean(error_MT_All,2),'-ro'); hold on;
plot(graph_node_num_set,mean(error_LS_All,2),'-bo');
plot(graph_node_num_set,mean(error_LS2_All,2),'-ko');
legend('MT','LS','LS2');
grid on;

figure; 
semilogy(graph_node_num_set,mean(error_MT_All,2),'-ro'); hold on;
semilogy(graph_node_num_set,mean(error_LS_All,2),'-bo');
semilogy(graph_node_num_set,mean(error_LS2_All,2),'-ko');
legend('MT','LS','LS2');
grid on;
