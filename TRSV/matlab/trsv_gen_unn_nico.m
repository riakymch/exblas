function trsv_gen_unn_nico()
  %#codegen
  n = 20;
  c = 20 * sort(rand(n,1));
  for i = 1:n
    [A, b] = trsv_gen_nico(n, c(i));
    i 
  end
end

function [A, b] = trsv_gen_nico(n, c)
    A = zeros(n);
    b = zeros(n, 1);

    % Si n est impair, on se ramene au cas pair
    if ~mod(n,2)
        A(n,n) = 1.0;
        b(n,1) = 1.0;
        n = n-1;
    end
    p = (n+1)/2; % n = 2*p - 1

    %  On commence par generer A(1:p, 1:n) et b(1:p)
    D = diag((ones(1,p) - 2*rand(1,p)) .* linspace(10^(-c),10^(c),p));
    U = triu((1 - 2*rand(p)) .* (10.^round(c*(1 - 2*rand(p)))), 1);
    A(1:p,1:p) = D + U;
    % A l'aide de l'algorithme VecSum, on calcule A(1:p, p+1:n) et b(1:p)
    % de maniere a ce que l'on aie exactement sum(A(i:)) = b(i)
    for i = 1:p
        t = VecSum(A(i,1:p), p);
        A(i,p+1:2*p-1) = -t(1:p-1);
        b(i) = t(p);
    end

    % On genere maintenant A(p+1:n,p+1:n) et b(p+1:n)
    % A(p+1:n,p+1:n) est generee aleatoirement avec des coefficient 
    % compris entre -1 et 1
    A(p+1:n,p+1:n) = triu(ones(p-1) - 2*rand(p-1));
    % b(p+1:n) est le resultat du produit A(p+1:n,p+1:n) * ones(p-1,1)
    % calcule avec une grande precision
    %b(p+1:n) = sym(A(p+1:n,p+1:n), 'f') * sym(ones(p-1,1), 'f');
    %b(p+1:n) = double(b(p+1:n));
    b(p+1:n) = A(p+1:n,p+1:n) * ones(p-1,1);
end

function [x, y] = TwoSum(a,b)
    x = a + b;
    z = x - a;
    y = (a - (x - z)) + (b - z);
end

function t = VecSum(p, n)
    s = zeros(n,1);
    t = zeros(n,1);
    s(1) = p(1);
    for i = 2:n
        [s(i), t(i-1)] = TwoSum(s(i-1),p(i));

    end
    t(n) = s(n);
end