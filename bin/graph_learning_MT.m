function Lhat = graph_learning_MT(X,del_t)

N = size(X,1);
M = size(X,2) - 1;
N_process = size(X,3);

dX = diff(X,1,2)/del_t; % shapeL N x (M) x N_process
aX = 0.5*(X(:,1:M,:) + X(:,2:M+1,:)); 

Lhat = zeros(N);
for kk = 1:N
    Xkk = [];
    Gkk = [];
    for ii = 1:N_process
        Xkk = [Xkk, dX(kk,:,ii)];
        Gkk = [Gkk, -aX(:,1:M,ii)]; 
    end
    %Lkk = mrdivide(Xkk,Gkk);   % give many warnings, depends on condition number
    Lkk = Xkk*pinv(Gkk);
    Lhat(kk,:) = Lkk;
end

end