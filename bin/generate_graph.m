function [A, D, L] = generate_graph(N,graph_seed)

rng(graph_seed);

% adjacency matrix
A = abs(randn(N));
A = triu(A,1);
A = A + transpose(A);

% degree matrix
D = sum(A,1);
D = diag(D);

% Laplacian matrix
L = D - A;

end