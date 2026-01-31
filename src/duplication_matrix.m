function D = duplication_matrix(N)
    p = N*(N+1)/2;
    D = zeros(N*N,p);
    k=1;
    for j=1:N
        for i=j:N
            row1 = i + (j-1)*N;
            D(row1,k)=1;
            if i~=j
                row2 = j + (i-1)*N;
                D(row2,k)=1;
            end
            k=k+1;
        end
    end
end
