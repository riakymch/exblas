function trsv_unn_sym()
  condA = [];
  err_d = [];
  err_k = [];
  
  n = 40;
  c = 20 * sort(rand(n,1));
  for i = 1:n
    %[A, b] = trsv_gen_unn_my(i);
    [A, b] = trsv_gen_unn_nico(n, c(i));

    [condA(i), err_d(i), err_k(i)] = trsv_unn_exact(n, A, b);
    i
  end

  err_d
  err_k
  condA

  %ax = plotyy(condA, err_d, condA, err_k);
  loglog(condA, err_d, '+', condA, err_k, 'o');
  %loglog(condA, err_k);
  %ax = plotyy(condA, err_d, condA, err_k, 'loglog');
  hold on;
  xlim([1, 10^50]);
  ylim([10^(-18), 10]);
  xlabel('CondA');
  ylabel('Error');
  legend('err_d','err_k');
  %grid on;
  xlims = get(gca,'XLim');
  ylims = get(gca,'YLim');
  u = 2^(-53);
  loglog([u^(-1) u^(-1)], ylims, '--k');
  hold on;
  loglog([u^(-2) u^(-2)], ylims, '--k');
  hold on;
  loglog(xlims, [u u], '--k');
  %ylabel(ax(2), 'Error Kulisch');
end

function x = trsv_unn_d(n, A, b)
  %trsv for unn matrices
  for i = n:-1:1
    s = b(i);
    for j = i+1:n
      s = s - A(i,j) * x(j);
    end
    x(i) = s / A(i, i);
  end
end

function x = trsv_unn_kulisch(n, A, b)
  b = sym(b);
  
  %trsv for unn matrices
  for i = n:-1:1
    s = b(i);
    for j = i+1:n
      s = s - sym(A(i,j)) * sym(x(j));
    end
    x(i) = double(s) / A(i, i);
  end
end

function [condA, err_d, err_k] = trsv_unn_exact(n, A, b)
  x_k = sym(trsv_unn_kulisch(n, A, b), 'd');
  x_d = sym(trsv_unn_d(n, A, b), 'd');
  
  A = sym(A);
  b = sym(b);
  %x_e = A \ b;
  for i = n:-1:1
    s = b(i);
    for j = i+1:n
      s = s - A(i,j) * x_e(j);
    end
    x_e(i) = s / A(i, i);
  end

  %compute error
  norm_e = max(double(abs(x_e)));
  err_k = max(double(abs(x_e - x_k))) / norm_e;
  err_d = max(double(abs(x_e - x_d))) / norm_e;
  
  %compute cond number
  %A_inv = inv(A);
  %condA = norminf_m(A, n) * norminf_m(A_inv, n);
  condA = condAx(A,x_e,n);
end

function res = condAx(A, x, n)
  A_inv = inv(A);
  y = abs(A_inv) * abs(A) * abs(x)';

  res = max(double(y)) / max(double(abs(x)));
end

function res = norminf_m(A, n)
  res = 0.0;

  for i = 1:n
      sum = sym(0.0, 'f');
      for j = 1:n
          sum = sum + abs(A(i,j));
      end
      sum_d = double(abs(sum));
      if res < sum_d
          res = sum_d;
      end
  end
end

function [A, b] = trsv_gen_unn_my(n)
    A = triu(rand(n));
    for i = 1:n
        A(i,i) = 10^((-1)^i*10*(i-1)/n);
    end  
    b = rand(n, 1);
end

function [x, y] = TwoSum(a,b)
    x = a + b;
    z = x - a;
    y = (a - (x - z)) + (b - z);
end

function t = VecSum(p, n)
    s(1) = p(1);
    for i = 2:n
        [s(i), t(i-1)] = TwoSum(s(i-1),p(i));
    end
    t(n) = s(n);
end

function [A, b] = trsv_gen_unn_nico(n, c)
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
    b(p+1:n) = sym(A(p+1:n,p+1:n), 'f') * sym(ones(p-1,1), 'f');
    b(p+1:n) = double(b(p+1:n));
end

