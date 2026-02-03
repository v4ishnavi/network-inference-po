function [A_hat, W_hat, C_hat, hist] = PO_model3( ...
    X_obs, beta, del_t, K, max_iter, tol, A_full_true)
    
    [N, T, N_proc] = size(X_obs);
    A_true = A_full_true(1:N, 1:N);
    W_true = A_full_true(1:N, N+1:end); 
    eps_val = 1e-6;
    
    Y_list = {}; X_mid_list = {};
    for pp = 1:N_proc
        dX = diff(X_obs(:,:,pp), 1, 2) / del_t;
        X_m = 0.5 * (X_obs(:,1:end-1,pp) + X_obs(:,2:end,pp));
        
        % MASK: Ignore points where nodes are saturated (x > 0.98)
        mask = all(X_m < 0.98, 1); 
        
        denom = beta * (1 - X_m(:, mask) + eps_val);
        Y_list{pp} = dX(:, mask) ./ denom;
        X_mid_list{pp} = X_m(:, mask);
    end
    
    Y_stack = cell2mat(Y_list);   % N x S
    X_stack = cell2mat(X_mid_list); % N x S
    S_valid = size(Y_stack, 2);
    
    % ---------- 2. Initialization ----------
    D = duplication_matrix(N);
    p = size(D,2);
    
    % Build global Phi for A_OO update
    Phi_A_global = zeros(N*S_valid, p);
    for s = 1:S_valid
        rows = (s-1)*N + 1 : s*N;
        Phi_A_global(rows, :) = (kron(X_stack(:,s)', eye(N)) * D);
    end

    % Step A: Initial Naive A_OO
    opts = optimoptions('lsqlin', 'Display', 'off');
    a0 = lsqlin(Phi_A_global, Y_stack(:), [], [], [], [], zeros(p,1), [], [], opts);
    A = vech_to_sym(a0, N);

    fprintf('A init error %.2f \n', 100*norm(A - A_true, 'fro') / norm(A_true, 'fro'));
    
    % Step B: SVD on the Interference Operator L
    % R approx L * X_stack  => L is the mapping from observed states to residuals
    R = Y_stack - A * X_stack;
    L_hat = R * X_stack' / (X_stack * X_stack' + 1e-3*eye(N));
    
    [U, S_mat, V] = svds(L_hat, K);
    
    % Ensure U is mostly positive for SI weights
    for k = 1:K
        if sum(U(:,k)) < 0, U(:,k) = -U(:,k); V(:,k) = -V(:,k); end
    end
    
    % W is N x K, C is K x N
    W = max(1e-3, U * sqrt(S_mat)); 
    C = max(1e-3, sqrt(S_mat) * V'); % V' is already K x N

    % ---------- 3. Alternating Minimization ----------
    hist.obj = []; hist.dA = []; hist.dW = [];
    
    for it = 1:max_iter
        % --- Update A_OO (Fix W, C) ---
        Y_clean = Y_stack - (W * C) * X_stack;
        a_vec = lsqlin(Phi_A_global, Y_clean(:), [], [], [], [], zeros(p,1), ones(p,1), [], opts);
        A_new = vech_to_sym(a_vec, N);
        
        % --- Update Mapping C (Fix A, W) ---
        % Solve: min || R - W*C*X ||^2 
        R = Y_stack - A_new * X_stack;
        lambda = 1e-6;
        WtW = W' * W + lambda * eye(K);
        XtX = X_stack * X_stack' + lambda * eye(N);
        
        % Solution for C: (W'W)^-1 * (W' R X') * (XX')^-1
        C_target = (WtW \ (W' * R * X_stack')) / XtX;
        C_new = 0.7*C + 0.3*max(0, C_target);

        % --- Update Influence W (Fix A, C) ---
        Z_eff = C_new * X_stack; % Predicted hidden states
        W_new = zeros(N, K);
        for i = 1:N
            W_new(i, :) = lsqlin(Z_eff', R(i, :)', [], [], [], [], zeros(K,1), ones(K,1), [], opts)';
        end
        
        % --- Normalization (Force C*x into 0-1 range) ---
        for k = 1:K
            z_max = max(C_new(k, :) * X_stack);
            if z_max > 1e-3
                C_new(k, :) = C_new(k, :) / z_max;
                W_new(:, k) = W_new(:, k) * z_max;
            end
        end

        % --- Metrics & Convergence ---
        curr_obj = norm(Y_stack - (A_new * X_stack + W_new * C_new * X_stack), 'fro')^2 / S_valid;
        
        if ~isempty(hist.obj) && curr_obj > hist.obj(end)
            % Damping if objective rises
            A = 0.9*A + 0.1*A_new;
            W = 0.9*W + 0.1*W_new;
            C = 0.9*C + 0.1*C_new;
        else
            A = A_new; W = W_new; C = C_new;
        end
        
        hist.obj(it) = curr_obj;
        hist.dA(it) = norm(A - A_true, 'fro') / (norm(A_true, 'fro') + 1e-9);
        hist.dW(it) = norm(W - W_true, 'fro') / (norm(W_true, 'fro') + 1e-9);
        
        if mod(it, 10) == 0 || it == 1
            fprintf('Iter %3d: MSE=%.3e, ErrA=%.2f%%, ErrW=%.2f%%\n', ...
                it, curr_obj, hist.dA(it)*100, hist.dW(it)*100);
        end
        
        if it > 1 && abs(hist.obj(it) - hist.obj(it-1))/hist.obj(it-1) < tol
            break;
        end
    end
    
    A_hat = A; W_hat = W; C_hat = C;
end

function A = vech_to_sym(v, n)
    A = zeros(n,n);
    idx = 1;
    for j = 1:n
        for i = j:n
            A(i,j) = v(idx); A(j,i) = v(idx);
            idx = idx + 1;
        end
    end
    A = A - diag(diag(A));
end

function D = duplication_matrix(n)
    m = n*(n+1)/2;
    D = zeros(n*n, m);
    count = 1;
    for j = 1:n
        for i = j:n
            res = zeros(n,n);
            res(i,j) = 1; res(j,i) = 1;
            D(:, count) = res(:);
            count = count + 1;
        end
    end
end