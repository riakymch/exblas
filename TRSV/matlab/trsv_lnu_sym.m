function trsv_lnu_sym()
  condA = [];
  err_d = [];
  err_k = [];
  
  j=1;
  
  %lnu
  alpha=224;
  for i = 1:21
    [A, b] = trsv_gen_lnu(alpha, i);
    
    [condA(j), err_d(j), err_k(j)] = trsv_lnu_exact(i, A, b);
    if err_d(j) > 1.0
        err_d(j) = 1.0;
    end
    if err_k(j) > 1.0
        err_k(j) = 1.0;
    end
    j = j+1;
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
  u = 2^(-53);
  xlims = get(gca,'XLim');
  ylims = get(gca,'YLim');
  loglog([u^(-1) u^(-1)], ylims, '--k');
  hold on;
  loglog([u^(-2) u^(-2)], ylims, '--k');
  hold on;
  loglog(xlims, [u u], '--k');
  %ylabel(ax(2), 'Error Kulisch');
end

function x = trsv_lnu_d(n, A, b)
  for i = 1:n
    s = b(i);
    for j = 1:i-1
      s = s - A(i,j) * x(j);
    end
    x(i) = s / A(i, i);
  end
end

function x = trsv_lnu_kulisch(n, A, b)
  b = sym(b);

  for i = 1:n
    s = b(i);
    for j = 1:i-1
        s = s - sym(A(i,j)) * sym(x(j));
    end
    x(i) = double(s) / A(i, i);
  end
end

function [condA, err_d, err_k] = trsv_lnu_exact(n, A, b)
  %double
  x_k = sym(trsv_lnu_kulisch(n, A, b), 'f');
  x_d = sym(trsv_lnu_d(n, A, b), 'f');

  A = sym(A);
  b = sym(b);
  %x_e = A \ b;
  for i = 1:n
    s = b(i);
    for j = 1:i-1
      s = s - A(i,j) * x_e(j);
    end
    x_e(i) = s / A(i, i);
  end

  %compute error
  norm_x_e = max(double(abs(x_e)));
  err_k = max(double(abs(x_e - x_k))) / norm_x_e;
  err_d = max(double(abs(x_e - x_d))) / norm_x_e;
  
  %compute cond number
  %condA = norminf_m(A, n) * norminf_m(inv(A), n);
  condA = condAx(A,x_e,n);
end

function res = condAx(A, x, n)
  A_inv = inv(A);
  y = abs(A_inv) * abs(A) * abs(x)';

  res = max(double(abs(y))) / max(double(abs(x)));
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

function [A, b] = trsv_gen_lnu(alpha, n)
    A = zeros(n,n);
    
    A(1,1) = 100;
    for i = 2:n
        A(i,i) = 1;
    end
    for i = 2:n
        for j = 1:i-1
            A(i,j) = (-1)^(i+j) * alpha;
        end
    end
    
    b = zeros(n, 1);
    b(1) = 1;
    for i = 2:n
        b(i) = -((alpha+1)/100) * (-2)^(i-2);
    end    
end
