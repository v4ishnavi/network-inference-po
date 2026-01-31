
clear;

binary_graph = 0;
add_noise = 1;

% data sampling
T = 100;
del_t = 0.001;
t = 0:del_t:T*del_t;

Num_graphs = 100;
N = 10;

N_process_All = 1:10;

error_MT_All = zeros(numel(N_process_All),Num_graphs);
error_LS_All = zeros(numel(N_process_All),Num_graphs);
error_LS2_All = zeros(numel(N_process_All),Num_graphs);

parpool(4);

for iProcess = 1:numel(N_process_All)

    disp(iProcess);

    N_process = N_process_All(iProcess);

    %% graph generation
    parfor iGraph = 1:Num_graphs
        graph_seed = iGraph;

        if binary_graph == 0
            [A, D, L] = generate_graph(N,graph_seed);
        else
            [A, D, L] = generate_graph_binary(N,graph_seed);
        end

        %% data generation
        data_seed = 100 + iGraph;
        
        X = [];
        for ii = 1:N_process
            new_data = generate_heat_diffusion(L,t,data_seed+ii);
            X(:,:,ii) = new_data;
        end

        % add noise
        if add_noise == 1
            noise_sigma = 10e-4;
            X = X + noise_sigma*randn(size(X));
        end

        %% graph learning
        %Lhat_MT = graph_learning_MT(X,del_t);
        Lhat_LS = graph_learning_proposed_LS(X,del_t);
        Lhat_LS2 = graph_learning_proposed_LS2(X,del_t);

        if binary_graph == 1
            %Lhat_MT = round(Lhat_MT);
            Lhat_LS = round(Lhat_LS);
            Lhat_LS2 = round(Lhat_LS2);
        end
        
        %Lerror_MT = 100*norm(Lhat_MT-L,"fro")/norm(L,"fro");
        %error_MT_All(iProcess,iGraph) = Lerror_MT;

        Lerror_LS = 100*norm(Lhat_LS-L,"fro")/norm(L,"fro");
        error_LS_All(iProcess,iGraph) = Lerror_LS;

        Lerror_LS2 = 100*norm(Lhat_LS2-L,"fro")/norm(L,"fro");
        error_LS2_All(iProcess,iGraph) = Lerror_LS2;

    end
end

delete(gcp('nocreate'));

figure; 
hold on;
%plot(N_process_All,mean(error_MT_All,2),'-ro');
plot(N_process_All,mean(error_LS_All,2),'ro-','linewidth',1,'MarkerSize',10);
plot(N_process_All,mean(error_LS2_All,2),'bs-.','linewidth',1,'MarkerSize',10);
ylim([0 100]);
%legend('MT','LS','LS2');
legend('Least squares','Least squares plus');
grid on; box on;
set(gca, 'YScale', 'log');
xlabel('# number of processes'); ylabel('Average relative error %');
fontsize(scale=1.5);
