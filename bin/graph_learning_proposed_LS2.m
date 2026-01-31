function Lhat = graph_learning_proposed_LS2(X,del_t)

N = size(X,1);
M = size(X,2) - 1;
N_process = size(X,3);
disp("[DEBUG]: N, M, N_process");
dX = diff(X,1,2)/del_t;
aX = 0.5*(X(:,1:M,:) + X(:,2:M+1,:));

U = []; v = [];
for ii = 1:N_process
    for tt = 1:M
        Ut = [];
        for kk = 1:N
            Ut = [Ut, [zeros(kk-1,N-kk+1); aX(kk:N,tt,ii)'; [zeros(N-kk,1) aX(kk,tt,ii)*eye(N-kk)]]];
        end
        U = [U; Ut];
    end
    temp = -dX(:,:,ii);
    v = [v; temp(:)];
end
disp("[DEBUG]: Done with U and v assembly");
disp("[DEBUG]: Imposing no self-loops constraint");
% add equations to impose no self loops (i.e., Laplacian row sum is zero)
Ns = round(N*(N+1)/2);
U1 = zeros(N,Ns);
v1 = zeros(N,1);

idx1 = 1;
for kk = 1:N
    idx2 = idx1 + N - kk;
    
    temp = N-1:-1:N-kk+1;
    idx3 = [idx1+1:idx2, kk, kk + cumsum(temp)];

    U1(kk,idx3) = 1;
    idx1 = idx2 + 1;
end

U = [U; U1];
v = [v; v1];

%disp(strcat('required rank = ',num2str(Ns)));
%disp(strcat('current rank = ',num2str(rank(U))));

Lvec = pinv(U)*v;

Lhat = zeros(N);
idx = 1;
disp("[DEBUG]: Reshaping Lvec to Lhat...");
for kk = 1:N
    Lhat(kk,kk:N) = Lvec(idx:idx+N-kk);
    Lhat(kk:N,kk) = Lvec(idx:idx+N-kk);
    idx = idx + N - kk + 1;
end

%figure; plot(dX'); grid on;

end