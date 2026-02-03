del_t = evalin('base','del_t');   % if del_t exists

X_mid_all = 0.5 * ( ...
    X_obs_all(:,1:end-1,:) + X_obs_all(:,2:end,:) );


pp = 1;                    % choose ONE observed process
K  = size(C_hat, 1);       % number of hidden processes
T1 = size(X_mid_all, 2);   % T-1

Z = zeros(K, T1);

for t = 1:T1
    xt = X_mid_all(:, t, pp);   % N x 1
    Z(:, t) = C_hat * xt;       % K x 1
end

figure;
for k = 1:K
    subplot(K,1,k)
    plot(Z(k,:), 'LineWidth', 1.5)
    grid on
    ylabel(sprintf('z_%d(t)', k))
    
    if k == 1
        title(sprintf('Hidden dynamics: C x_t (process %d)', pp))
    end
    if k == K
        xlabel('time index t')
    end
end
