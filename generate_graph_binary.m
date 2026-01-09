function [A, D, L] = generate_graph_binary(N,graph_seed)

rng(graph_seed);

% adjacency matrix
A = abs(randn(N));
A = triu(A,1);
A = A + transpose(A);

% make binary
A = A > 0.2;

% degree matrix
D = sum(A,1);
D = diag(D);

% Laplacian matrix
L = D - A;

end