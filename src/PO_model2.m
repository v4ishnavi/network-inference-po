function [A_hat, W_hat, Z_hat, hist] = PO_model2( ...
    X_obs, beta, del_t, K, max_iter, tol, A_full_true)
    
    % SI_PARTIAL_OBS_INFERENCE
    [N, T, N_proc] = size(X_obs);
    A_true = A_full_true(1:N, 1:N);
    W_true = A_full_true(1:N, N+1:end);
    S = (T-1) * N_proc;
    eps_val = 1e-8;
    
    % ---------- Precompute Derivatives and Midpoints ----------
    dX_all = zeros(N, T-1, N_proc);
    X_mid_all = zeros(N, T-1, N_proc);
    for pp = 1:N_proc
        dX_all(:,:,pp) = diff(X_obs(:,:,pp), 1, 2) / del_t;
        X_mid_all(:,:,pp) = 0.5 * (X_obs(:,1:end-1,pp) + X_obs(:,2:end,pp));
    end
    
    % ---------- Initialization ----------
    D = duplication_matrix(N);
    p = size(D,2);
    
    % Step 0: Initial LS estimate for A (using lsqlin instead of pinv)
    U_init = []; Y_init = [];
    for pp = 1:N_proc
        for tt = 1:(T-1)
            xt = X_mid_all(:,tt,pp);
            if max(xt) > 0.98, continue; end % Avoid saturated regions for numerical stability
            wt = beta * (1 - xt + eps_val);
            phi_t = diag(wt) * (kron(xt', eye(N)) * D);
            U_init = [U_init; phi_t];
            Y_init = [Y_init; dX_all(:,tt,pp)];
        end
    end

    % bounds: non-negative symmetric upper-triangular entries
    lb_a = zeros(p,1);
    opts = optimoptions('lsqlin', 'Display', 'off', 'MaxIter', 2000);
    a0 = lsqlin(U_init, Y_init, [], [], [], [], lb_a, [], [], opts);

    A = zeros(N,N); idx = 1;
    for j = 1:N, for k = j:N
        A(j,k) = a0(idx); A(k,j) = a0(idx); idx = idx + 1;
    end; end
    A = max(0, A - diag(diag(A)));

    fprintf('A init error %.2f \n', 100*norm(A - A_true, 'fro') / norm(A_true, 'fro'));

    % Step 1: Initialize W and Z using SVD on residuals
    R_init = zeros(N, S); col = 1;
    for pp = 1:N_proc
        for tt = 1:(T-1)
            wt = beta * (1 - X_mid_all(:,tt,pp) + eps_val);
            R_init(:,col) = (dX_all(:,tt,pp) - wt .* (A * X_mid_all(:,tt,pp))) ./ (wt + eps_val);
            col = col + 1;
        end
    end
    
    % SVD low-rank approximation
    [U, S_mat, V] = svd(R_init, 'econ');
    sqrtSk = sqrt(S_mat(1:K, 1:K));
    
    W = U(:, 1:K) * sqrtSk;
    Z = sqrtSk * V(:, 1:K)';
    
    % Rectify for SI Dynamics: Non-negativity is a physical requirement
    W = max(0, W); 
    Z = max(0, Z);
    
    % ---------- EM / Alternating Loop ----------
    hist.obj = []; hist.dA = []; hist.dW = [];
    patience = 5;
    
    % Reuse lsqlin options for E-step and M-step
    opts = optimoptions('lsqlin', 'Display', 'off', 'MaxIter', 2000);
    
    for it = 1:max_iter
        Z_prev = Z;
        
        % ----- E-step: Update Z given A and W (use lsqlin in place of pinv) -----
        col = 1;
        lb_z = zeros(K,1); ub_z = []; % z >= 0
        for pp = 1:N_proc
            for tt = 1:(T-1)
                xt = X_mid_all(:,tt,pp);
                dxt = dX_all(:,tt,pp);
                wt = beta * (1 - xt + eps_val);
                
                % Solve: dxt = wt .* (A*xt + W*zt) 
                % Target: dxt - wt.*(A*xt) = (wt.*W) * zt
                target = dxt - wt .* (A * xt);
                operator = diag(wt) * W; % N x K

                % If operator or target are all zeros, return zeros
                if norm(operator,'fro') < 1e-12 || norm(target) < 1e-12
                    Z(:,col) = zeros(K,1);
                else
                    Z(:,col) = lsqnonneg(operator, target);
                end

                col = col + 1;
            end
        end

        % ----- M-step: Update A and W jointly using Weighted LS lsqlin -----
        N_params = p + N*K;
        Y_stack = zeros(N*S, 1);
        Phi = zeros(N*S, N_params);
        row_idx = 1; col = 1;
        
        for pp = 1:N_proc
            for tt = 1:(T-1)
                xt = X_mid_all(:,tt,pp);
                zt = Z(:,col);
                wt = beta * (1 - xt + eps_val);
                
                % Weighted operators to maintain numerical stability
                Phi_A = diag(wt) * (kron(xt', eye(N)) * D);
                Phi_W = diag(wt) * kron(zt', eye(N));
                
                rows = row_idx : row_idx+N-1;
                Phi(rows, :) = [Phi_A, Phi_W];
                Y_stack(rows) = dX_all(:,tt,pp);
                
                row_idx = row_idx + N;
                col = col + 1;
            end
        end

        % Constrained Solve: 0 <= A, W <= inf
        lb = zeros(N_params, 1);
        opts = optimoptions('lsqlin', 'Display', 'off', 'MaxIter', 2000);
        lambda = 1e-6;
        Phi_aug = [Phi; sqrt(lambda)*eye(N_params)];
        Y_aug   = [Y_stack; zeros(N_params,1)];
        theta = lsqlin(Phi_aug, Y_aug, [], [], [], [], lb, [], [], opts);


        % Extract and Rebuild
        a_vec = theta(1:p);
        w_vec = theta(p+1:end);
        
        A_new = zeros(N,N); idx = 1;
        for j = 1:N, for k = j:N
            A_new(j,k) = a_vec(idx); A_new(k,j) = a_vec(idx); idx = idx + 1;
        end; end
        A_new = A_new - diag(diag(A_new));
        W_new = reshape(w_vec, [N,K]);

        % New step for Normalizing columsn of W to prevent the scaling ambiguity
        colnorms = sqrt(sum(W_new.^2, 1)) + eps;
        W_new = W_new ./ colnorms;
        Z = Z .* colnorms';

        % ----- Metrics and Convergence -----
        curr_obj = norm(Y_stack - Phi*theta)^2;
        hist.obj(it) = curr_obj;
        
        % Monitoring (Truth only for plotting, not used for decisions)
        hist.dA(it) = norm(A_new - A_true, 'fro') / norm(A_true, 'fro');
        hist.dW(it) = norm(W_new - W_true, 'fro') / norm(W_true, 'fro');
        
        fprintf('Iter %2d: Obj=%.3e, dA_truth=%.2f%%, dW_truth=%.2f%%\n', ...
            it, curr_obj, hist.dA(it)*100, hist.dW(it)*100);

        % Convergence Logic: Flatness or rising objective (overfitting)
        if it > 1
            rel_change = abs(hist.obj(it) - hist.obj(it-1)) / hist.obj(it-1);
            if rel_change < tol
                A_hat = A_new; W_hat = W_new; Z_hat = Z;
                fprintf('Converged (Objective Saturated).\n'); break;
            end
            
            % If objective increases for 5 consecutive steps, stop (overfitting)
            if it > patience && all(diff(hist.obj(end-patience:end)) > 0)
                A_hat = A_new; W_hat = W_new; Z_hat = Z;
                fprintf('Stopped (Objective rising/overfitting).\n'); break;
            end
        end

        A = A_new; W = W_new;
    end
    % If the loop finished without hitting a 'break', 
    % assign the current values to the output variables.
    if ~exist('A_hat', 'var')
        A_hat = A;
        W_hat = W;
        Z_hat = Z;
    end
end
