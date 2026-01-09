function X = generate_heat_diffusion(L,t,data_seed)
% returns heat diffusion data X given graph Laplacian L and time vector t
N = size(L,1);
Nt = length(t);

rng(data_seed);
x0 = randn(N,1);

[V,D] = eig(L);
D = diag(D);
Dt = D * t;
Dte = exp(-Dt);

X = zeros(N,Nt);

for ii = 1:Nt
    X(:,ii) = (V*diag(Dte(:,ii))/V)*x0;
end

end