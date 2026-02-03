function [A_hat, W_hat, C_hat_all, hist] = PO_model4( ...
    X_obs, beta, del_t, K, max_iter, tol, A_full_true)
    
    % SI_PARTIAL_OBS_ROBUST
    % Features: Process-Dependent Mapping + Trust-Region Stability check
    
    [N, T, N_proc] = size(X_obs);
    A_true = A_full_true(1:N, 1:N);
    W_true = A_full_true(1:N, N+1:end); 
    eps_val = 1e-8;
    
    % ---------- 1. Preprocess Data ----------
    Y_cell = cell(1, N_proc);
    X_cell = cell(1, N_proc);
    
    for pp = 1:N_proc
        dX = diff(X_obs(:,:,pp), 1, 2) / del_t;
        X_m = 0.5 * (X_obs(:,1:end-1,pp) + X_obs(:,2:end,pp));
        mask = all(X_m < 0.98, 1); 
        denom = beta * (1 - X_m(:, mask) + eps_val);
        Y_cell{pp} = dX(:, mask) ./ denom;
        X_cell{pp} = X_m(:, mask);
    end
    Y_stack = cell2mat(Y_cell);
    X_stack = cell2mat(X_cell);
    S_valid = size(Y_stack, 2);
    
    % ---------- 2. Initialization ----------
    D = duplication_matrix(N);
    p_dim = size(D,2);
    
    % Build global Phi for A_OO
    Phi_A_global = zeros(N*S_valid, p_dim);
    for s = 1:S_valid
        rows = (s-1)*N + 1 : s*N;
        Phi_A_global(rows, :) = (kron(X_stack(:,s)', eye(N)) * D);
    end

    % Init A (Global)
    opts = optimoptions('lsqlin', 'Display', 'off');
    lb_a = zeros(p_dim, 1);
    a0 = lsqlin(Phi_A_global, Y_stack(:), [], [], [], [], lb_a, [], [], opts);
    A = vech_to_sym(a0, N);
    
    % Init W, C (SVD on Residuals)
    R = Y_stack - A * X_stack;
    L_hat = R * X_stack' / (X_stack * X_stack' + 1e-3*eye(N)); 
    [U, S_mat, V] = svds(L_hat, K);
    
    % Sign convention
    for k = 1:K
        if sum(U(:,k)) < 0, U(:,k) = -U(:,k); V(:,k) = -V(:,k); end
    end
    W = max(1e-3, U * sqrt(S_mat)); 
    C_global = max(1e-3, sqrt(S_mat) * V');
    C_all = repmat(C_global, [1, 1, N_proc]);

    fprintf('Init: A err %.2f%%\n', 100*norm(A - A_true, 'fro')/norm(A_true,'fro'));

    % ---------- 3. Robust Alternating Loop ----------
    hist.obj = []; hist.dA = []; hist.dW = [];
    
    % Config
    curr_damping = 0.2; % Start cautious (0.0 = fast, 1.0 = frozen)
    lambda_reg = 1e-3;
    alpha_anchor = 5.0; 
    
    % Calculate initial objective
    prev_obj = calc_global_obj(Y_cell, X_cell, A, W, C_all);
    fprintf('Initial Obj: %.4e\n', prev_obj);

    for it = 1:max_iter
        % Save state in case we need to reject
        A_old = A; W_old = W; C_old = C_all;
        
        % --- Step 1: Candidate A Update ---
        Y_clean_cell = cell(1, N_proc);
        for pp = 1:N_proc
            Y_clean_cell{pp} = Y_cell{pp} - (W * C_all(:,:,pp) * X_cell{pp});
        end
        Y_clean_stack = cell2mat(Y_clean_cell);
        a_vec = lsqlin(Phi_A_global, Y_clean_stack(:), [], [], [], [], lb_a, [], [], opts);
        A_cand = vech_to_sym(a_vec, N);
        
        % --- Step 2: Candidate C Update (Anchored) ---
        C_cand = C_all;
        C_mean = mean(C_all, 3);
        WtW = W' * W + lambda_reg * eye(K); 
        
        for pp = 1:N_proc
            R_p = Y_cell{pp} - A_cand * X_cell{pp}; % Use A_cand!
            XtX = X_cell{pp} * X_cell{pp}' + lambda_reg * eye(N);
            C_data = (WtW \ (W' * R_p * X_cell{pp}')) / XtX;
            C_target = (1/(1+alpha_anchor))*C_data + (alpha_anchor/(1+alpha_anchor))*C_mean;
            C_cand(:,:,pp) = max(0, C_target);
        end
        
        % --- Step 3: Candidate W Update ---
        Z_stack_list = {}; Y_resid_list = {};
        for pp = 1:N_proc
            Z_stack_list{end+1} = C_cand(:,:,pp) * X_cell{pp}; % Use C_cand!
            Y_resid_list{end+1} = Y_cell{pp} - A_cand * X_cell{pp};
        end
        Z_stack = cell2mat(Z_stack_list);
        Y_resid = cell2mat(Y_resid_list);
        
        W_cand = zeros(N, K);
        for i = 1:N
            W_cand(i, :) = lsqlin(Z_stack', Y_resid(i, :)', [], [], [], [], zeros(K,1), [], [], opts)';
        end
        
        % Normalize Candidates
        colnorms = sqrt(sum(W_cand.^2, 1)) + eps;
        W_cand = W_cand ./ colnorms;
        for pp = 1:N_proc, C_cand(:,:,pp) = C_cand(:,:,pp) .* colnorms'; end

        % --- Step 4: TRUST REGION CHECK ---
        curr_obj = calc_global_obj(Y_cell, X_cell, A_cand, W_cand, C_cand);
        
        if curr_obj < prev_obj
            % ACCEPT
            A = (1-curr_damping)*A_cand + curr_damping*A_old;
            W = (1-curr_damping)*W_cand + curr_damping*W_old;
            C_all = (1-curr_damping)*C_cand + curr_damping*C_old;
            
            % Relax damping (go faster next time)
            curr_damping = max(0.05, curr_damping * 0.9);
            status = 'ACC';
            prev_obj = curr_obj; % Update baseline
        else
            % REJECT (Stability Kick-in)
            % Do NOT update baseline obj.
            % Take a tiny step towards candidate to escape saddle, but mostly stay put.
            safe_step = 0.05; 
            A = (1-safe_step)*A_old + safe_step*A_cand;
            W = (1-safe_step)*W_old + safe_step*W_cand;
            C_all = (1-safe_step)*C_old + safe_step*C_cand;
            
            % Increase damping (slow down next time)
            curr_damping = min(0.8, curr_damping * 1.5);
            status = 'REJ';
            % Recalculate obj for logging (it will be slightly different due to safe step)
            curr_obj = calc_global_obj(Y_cell, X_cell, A, W, C_all);
        end

        % Metrics
        hist.obj(it) = curr_obj;
        hist.dA(it) = norm(A - A_true, 'fro') / norm(A_true, 'fro');
        hist.dW(it) = norm(W - W_true, 'fro') / norm(W_true, 'fro');
        
        if mod(it, 10) == 0 || it == 1
            fprintf('It %3d [%s]: Obj=%.3e (Damp=%.2f) A_err=%.1f%%\n', ...
                it, status, curr_obj, curr_damping, hist.dA(it)*100);
        end
        
        % Check convergence
        if it > 10 && abs(hist.obj(it) - hist.obj(it-5))/hist.obj(it-5) < tol
             fprintf('Converged.\n'); break;
        end
    end
    
    A_hat = A; W_hat = W; C_hat_all = C_all;
end

function obj = calc_global_obj(Y_c, X_c, A, W, C)
    err = 0;
    count = 0;
    for pp = 1:length(Y_c)
        est = A * X_c{pp} + W * C(:,:,pp) * X_c{pp};
        err = err + norm(Y_c{pp} - est, 'fro')^2;
        count = count + numel(Y_c{pp});
    end
    obj = err / count;
end

% --- Helpers ---
function A = vech_to_sym(v, n)
    A = zeros(n,n); idx = 1;
    for j = 1:n, for i = j:n, A(i,j)=v(idx); A(j,i)=v(idx); idx=idx+1; end; end
    A = A - diag(diag(A));
end

function D = duplication_matrix(n)
    m = n*(n+1)/2; D = zeros(n*n, m); count = 1;
    for j = 1:n, for i = j:n, res=zeros(n,n); res(i,j)=1; res(j,i)=1; D(:,count)=res(:); count=count+1; end; end
end