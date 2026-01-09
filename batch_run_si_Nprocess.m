clear;

%% Parameters
binary_graph = 0;
add_noise = 1;

% SI parameters
beta = 1.5;
T_end = 2.0;
del_t = 0.05;
t = 0:del_t:T_end;

Num_graphs = 50;
N = 8;  % smaller graphs for SI

N_process_All = 1:2:7;

error_MT_All = zeros(numel(N_process_All), Num_graphs);
error_LS_All = zeros(numel(N_process_All), Num_graphs);
error_LS2_All = zeros(numel(N_process_All), Num_graphs);

parpool(4);

for iProcess = 1:numel(N_process_All)
    
    disp(['Process ', num2str(iProcess), '/', num2str(numel(N_process_All))]);
    
    N_process = N_process_All(iProcess);
    
    %% graph generation
    parfor iGraph = 1:Num_graphs
        graph_seed = iGraph;
        
        if binary_graph == 0
            [~, ~, L] = generate_graph(N, graph_seed);
        else
            [~, ~, L] = generate_graph_binary(N, graph_seed);
        end
        
        % Convert to adjacency for SI
        A_true = -(L - diag(diag(L)));
        A_true = max(0, A_true);
        A_true = 0.5 * (A_true + A_true');
        
        %% data generation
        data_seed = 100 + iGraph;
        
        X = [];
        for ii = 1:N_process
            new_data = generate_si_dynamics(A_true, t, beta, data_seed+ii);
            X(:,:,ii) = new_data;
        end
        
        % add noise
        if add_noise == 1
            noise_sigma = 0.01;
            X = X + noise_sigma*randn(size(X));
            X = max(0, min(1, X));  % keep in [0,1]
        end
        
        %% graph learning
        [Ahat_MT, ~] = graph_learning_si_MT(X, del_t, beta);
        [Ahat_LS, ~] = graph_learning_si_LS(X, del_t, beta);
        [Ahat_LS2, ~] = graph_learning_si_LS2(X, del_t, beta);
        
        if binary_graph == 1
            Ahat_MT = round(Ahat_MT);
            Ahat_LS = round(Ahat_LS);
            Ahat_LS2 = round(Ahat_LS2);
        end
        
        Aerror_MT = 100*norm(Ahat_MT-A_true,"fro")/norm(A_true,"fro");
        error_MT_All(iProcess,iGraph) = Aerror_MT;
        
        Aerror_LS = 100*norm(Ahat_LS-A_true,"fro")/norm(A_true,"fro");
        error_LS_All(iProcess,iGraph) = Aerror_LS;
        
        Aerror_LS2 = 100*norm(Ahat_LS2-A_true,"fro")/norm(A_true,"fro");
        error_LS2_All(iProcess,iGraph) = Aerror_LS2;
        
    end
end

delete(gcp('nocreate'));

figure; 
hold on;
plot(N_process_All, mean(error_MT_All,2), 'ro-', 'linewidth', 1, 'MarkerSize', 10);
plot(N_process_All, mean(error_LS_All,2), 'bs-', 'linewidth', 1, 'MarkerSize', 10);
plot(N_process_All, mean(error_LS2_All,2), 'k^-.', 'linewidth', 1, 'MarkerSize', 10);
ylim([0 100]);
legend('MT','LS','LS2 (Symmetric)');
grid on; box on;
set(gca, 'YScale', 'log');
xlabel('# number of processes'); 
ylabel('Average relative error %');
title('SI Dynamics: Error vs Number of Processes');
fontsize(scale=1.5);
