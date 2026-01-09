
% batch simulations for varying Noise & processes

clear;

binary_graph = 0;

% data sampling
T = 200;
del_t = 10e-3;
t = 0:del_t:T*del_t;

Num_graph_sims = 10;
Num_data_sims = 10;

N = 10;
N_process_set = 1:2:10;
noise_sigma_All = logspace(-5,-1,21);

error_MT_All = zeros(numel(noise_sigma_All),numel(N_process_set),Num_graph_sims,Num_data_sims);
error_LS_All = zeros(numel(noise_sigma_All),numel(N_process_set),Num_graph_sims,Num_data_sims);
error_LS2_All = zeros(numel(noise_sigma_All),numel(N_process_set),Num_graph_sims,Num_data_sims);

parpool(4);

for iNoise = 1:numel(noise_sigma_All)

    disp(strcat('noise = ',num2str(noise_sigma_All(iNoise))));
    noise_sigma = noise_sigma_All(iNoise);

    for iPS = 1:numel(N_process_set)

        disp(strcat('Nprocess = ',num2str(N_process_set(iPS))));
        N_process = N_process_set(iPS);

        % graph generation
        parfor iGraph = 1:Num_graph_sims

            graph_seed = iGraph;

            if binary_graph == 0
                [A, D, L] = generate_graph(N,graph_seed);
            else
                [A, D, L] = generate_graph_binary(N,graph_seed);
            end

            for iData = 1:Num_data_sims

                data_seed = 100*iGraph + iData;

                X = zeros(N,T+1,N_process);
                for ii = 1:N_process
                    new_data = generate_heat_diffusion(L,t,data_seed+ii);
                    X(:,:,ii) = new_data;
                end

                % add noise
                X = X + noise_sigma*randn(size(X));

                % graph learning
                Lhat_MT = graph_learning_MT(X,del_t);
                Lhat_LS = graph_learning_proposed_LS(X,del_t);
                Lhat_LS2 = graph_learning_proposed_LS2(X,del_t);

                if binary_graph == 1
                    Lhat_MT = round(Lhat_MT);
                    Lhat_LS = round(Lhat_LS);
                    Lhat_LS2 = round(Lhat_LS2);
                end

                Lerror_MT = 100*norm(Lhat_MT-L,"fro")/norm(L,"fro");
                error_MT_All(iNoise,iPS,iGraph,iData) = Lerror_MT;

                Lerror_LS = 100*norm(Lhat_LS-L,"fro")/norm(L,"fro");
                error_LS_All(iNoise,iPS,iGraph,iData) = Lerror_LS;

                Lerror_LS2 = 100*norm(Lhat_LS2-L,"fro")/norm(L,"fro");
                error_LS2_All(iNoise,iPS,iGraph,iData) = Lerror_LS2;

            end

        end

    end

end

delete(gcp('nocreate'));

save('results_Noise_Nprocess');
