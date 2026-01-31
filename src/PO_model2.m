function [A_hat, W_hat, Z_hat, hist] = PO_model1( ...
    X_obs, beta, del_t, K, max_iter, tol, method, A_full_true, init_mode)
% SI_PARTIAL_OBS_EM_MULTI
%
% INPUT:
%   X_obs      : (N x T x N_proc)   observed SI states
%   init_mode  : 'true' (use ground truth), 'random' (random init), 'ls' (least squares)
%
% OUTPUT:
%   A_hat  : (N x N)
%   W_hat  : (N x K)
%   Z_hat  : (K x S)   where S = (T-1) * N_proc
%   hist   : struct
% method -> abs, none 

    [N, T, N_proc] = size(X_obs);
    A_true = A_full_true(1:N, 1:N);
    W_true = A_full_true(1:N, N+1:end);
    
    % Total samples across all processes:
    S = (T-1) * N_proc;
    eps = 1e-8;
    
    % ---------- Compute derivatives and Ytilde for ALL processes ----------
    X_mid_all = zeros(N, T-1, N_proc);
    Ytilde_all = zeros(N, T-1, N_proc);

    for pp = 1:N_proc
        dX = diff(X_obs(:,:,pp), 1, 2) / del_t; % (N x T-1)
        X_mid = 0.5 * (X_obs(:,1:end-1,pp) + X_obs(:,2:end,pp));
        X_mid_all(:,:,pp) = X_mid;
        for tt = 1:(T-1)
            Ytilde_all(:,tt,pp) = dX(:,tt) ./ (beta*(1 - X_mid(:,tt)+eps));
        end
    end

    % ---------- Build duplication matrix (N^2 x p) ----------
    D = duplication_matrix(N);
    p = size(D,2);

    % ---------- INITIALIZATION: Half-vec LS for A ----------
    U = [];
    Y_A = [];
    saturation_threshold = 0.95; 
    for pp = 1:N_proc
        for tt = 1:(T-1)
            x_t = X_mid_all(:,tt,pp);

            if max(x_t) > saturation_threshold
             continue; 
            end

            y_t = Ytilde_all(:,tt,pp);
            Phi_A_t = kron(x_t', eye(N)) * D;  % (N x p)
            U   = [U;   Phi_A_t];
            Y_A = [Y_A; y_t];
        end
    end
    
    a0 = pinv(U) * Y_A;                 
    A0 = zeros(N,N);
    idx = 1;
    for j = 1:N
        for k = j:N
            A0(j,k) = a0(idx);
            A0(k,j) = a0(idx);
            idx = idx + 1;
        end
    end
    
    A0 = max(0,A0);
    A0 = A0 - diag(diag(A0));

    % Error after A0 initialization
    init_err_A = norm(A0 - A_true, 'fro') / norm(A_true, 'fro');
    fprintf('[INIT] Initial A_OO Relative Error: %.2f%%\n', init_err_A * 100);
    fprintf('----------------------------------------------------------------\n');

    % ---------- INITIALIZATION: NMF for W,Z ----------
    R = zeros(N, S);
    col = 1;
    for pp = 1:N_proc
        for tt = 1:(T-1)
            res = Ytilde_all(:,tt,pp) - A0 * X_mid_all(:,tt,pp);
            if method == "abs"
                R(:,col) = abs(res);
            else  % method == "none"
                R(:,col) = res;
            end
            col = col + 1;
        end
    end
    
    [~, S_R, ~] = svd(R, 'econ');
    sing_vals = diag(S_R);
    var_cum = cumsum(sing_vals.^2) / sum(sing_vals.^2);
    rank_95 = find(var_cum >= 0.95, 1);
    rank_99 = find(var_cum >= 0.99, 1);
    
    if isempty(rank_95), rank_95 = length(sing_vals); end
    if isempty(rank_99), rank_99 = length(sing_vals); end
    
    fprintf('[INFO] |R| rank for 95%% var: %d, 99%% var: %d (Chosen K=%d)\n', rank_95, rank_99, K);
    fprintf('[INFO] Top 5 singular values: ');
    fprintf('%.3e ', sing_vals(1:min(5, length(sing_vals))));
    fprintf('\n');
    
    R = abs(R);
    [W0, Z0] = nnmf(R, K);

    % Normalize columns of W0 to prevent scaling drift
    for kk = 1:K
        nrm = norm(W0(:,kk));
        if nrm > 1e-6
            W0(:,kk) = W0(:,kk) / nrm;
        else
            W0(:,kk) = 0; % Kill dead factors
        end
    end

    % ---------- INITIALIZATION MODES ----------
    if nargin < 9
        init_mode = 'ls';  % default
    end
    
    % SAFEGUARD: If user asks for 'true' but K doesn't match truth, fallback.
    if strcmp(init_mode, 'true') && size(W_true, 2) ~= K
        fprintf('[WARN] init_mode="true" requested but K=%d != N_hidden=%d. Fallback to "ls".\n', K, size(W_true, 2));
        init_mode = 'ls';
    end
    
    if strcmp(init_mode, 'true')
        A = A_true;
        W = W_true;
        Z = Z0; % We can't know true Z easily without inferring, so keep Z0
        fprintf('[INIT] Using ground truth A and W\n');
    elseif strcmp(init_mode, 'random')
        A = 0.1 * rand(N, N);
        A = 0.5 * (A + A');  % symmetric
        A = A - diag(diag(A));  % no self-loops
        A = max(0, A);
        W = W0;  % Use NMF initialization for W
        Z = Z0;  % Use NMF initialization for Z
        fprintf('[INIT] Using random A_oo with NMF W and Z\n');
    else  % 'ls' mode
        A = A0;
        W = W0;
        Z = Z0;
        fprintf('[INIT] Using least squares A and NMF W\n');
    end
    
    % ---------- EM LOOP ----------
    hist.A_hist = {};
    hist.W_hist = {};
    hist.Z_hist = {};
    hist.obj = [];
    hist.dA = [];
    hist.dW = [];
    hist.dZ = [];
    hist.R_norm = [];
    
    for it = 1:max_iter

        % ----- E-step -----
        Z_prev = Z;
        M_inv = inv(W'*W + 1e-6*eye(K)); % (K x K)

        col = 1;
        R_current = zeros(N, S);
        for pp = 1:N_proc
            for tt = 1:(T-1)
                r_t = Ytilde_all(:,tt,pp) - A * X_mid_all(:,tt,pp);
                R_current(:, col) = r_t;
                Z(:,col) = M_inv * (W' * r_t); % (K x 1)
                col = col + 1;
            end
        end

        % ----- M-step -----
        N_params = p + N*K;
        Y_stack = zeros(N*S,1);
        Phi     = zeros(N*S, N_params);
        row_offset = 0;
        col = 1;

        for pp = 1:N_proc
            for tt = 1:(T-1)
                y_t = Ytilde_all(:,tt,pp);
                x_t = X_mid_all(:,tt,pp);
                z_t = Z(:,col);
                
                Phi_A_t = kron(x_t', eye(N)) * D;     % (N x p)
                Phi_W_t = kron(z_t', eye(N));        % (N x N*K)
                Phi_t   = [Phi_A_t, Phi_W_t];        % (N x N_params)
                
                idx_rows = (row_offset+1):(row_offset+N);
                Phi(idx_rows,:)  = Phi_t;
                Y_stack(idx_rows)= y_t;
                row_offset = row_offset + N;
                col = col + 1;
            end
        end

        lambda = 1e-5;
        theta = (Phi'*Phi + lambda*eye(N_params)) \ (Phi'*Y_stack);

        % Extract a and w
        a_vec = theta(1:p);
        w_vec = theta(p+1:end);

        % Rebuild A
        A_new = zeros(N,N);
        idx = 1;
        for j = 1:N
            for k = j:N
                A_new(j,k)=a_vec(idx);
                A_new(k,j)=a_vec(idx);
                idx = idx + 1;
            end
        end
        A_new = max(0,A_new);
        A_new = A_new - diag(diag(A_new));

        % Rebuild W
        W_new = reshape(w_vec, [N,K]);
        W_new = max(0,W_new);

        hist.A_hist{it} = A_new;
        hist.W_hist{it} = W_new;
        hist.Z_hist{it} = Z;
        hist.obj(it)    = norm(Y_stack - Phi*theta)^2;

        % --- ERROR TRACKING (Safe for mismatched K) ---
        dA_truth = norm(A_new - A_true, 'fro') / norm(A_true, 'fro');
        
        if size(W_new) == size(W_true)
            dW_truth = norm(W_new - W_true, 'fro') / norm(W_true, 'fro');
        else
            dW_truth = NaN; % Cannot compare directly if dimensions differ
        end

        dA_change = norm(A_new - A, 'fro') / (norm(A, 'fro') + eps);
        dW_change = norm(W_new - W, 'fro') / (norm(W, 'fro') + eps);
        dZ = norm(Z - Z_prev, 'fro') / (norm(Z_prev, 'fro') + eps);
        R_norm = norm(R_current, 'fro');
        
        hist.dA(it) = dA_truth;  
        hist.dW(it) = dW_truth;
        hist.dZ(it) = dZ;
        hist.R_norm(it) = R_norm;
        
        if isnan(dW_truth)
             fprintf('Iter %2d: dA_truth=%.3e, dW_truth=N/A, dA_chg=%.3e, dW_chg=%.3e, dZ=%.3e, obj=%.3e\n', ...
            it, dA_truth, dA_change, dW_change, dZ, hist.obj(it));
        else
            fprintf('Iter %2d: dA_truth=%.3e, dW_truth=%.3e, dA_chg=%.3e, dW_chg=%.3e, dZ=%.3e, obj=%.3e\n', ...
            it, dA_truth, dW_truth, dA_change, dW_change, dZ, hist.obj(it));
        end

        A = A_new;
        W = W_new;

        if dA_change < tol
            fprintf('Converged.\n');
            break;
        end
    end

    A_hat = A;
    W_hat = W;
    Z_hat = Z;

end