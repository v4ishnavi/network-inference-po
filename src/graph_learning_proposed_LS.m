function Lhat = graph_learning_proposed_LS(X,del_t)

N = size(X,1);
M = size(X,2) - 1;
N_process = size(X,3);

dX = diff(X,1,2)/del_t;
aX = 0.5*(X(:,1:M,:) + X(:,2:M+1,:));

%U = zeros(N*M,Ns);
U = []; v = [];
disp("[DEBUG]: Assembling U and v matrices...");
for ii = 1:N_process
    for tt = 1:M
        %Ut = zeros(N,Ns);
        Ut = [];
        for kk = 1:N
            Ut = [Ut, [zeros(kk-1,N-kk+1); aX(kk:N,tt,ii)'; [zeros(N-kk,1) aX(kk,tt,ii)*eye(N-kk)]]];
        end
        U = [U; Ut];
    end
    temp = -dX(:,:,ii);
    v = [v; temp(:)];
end

%Ns = round(N*(N+1)/2);
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