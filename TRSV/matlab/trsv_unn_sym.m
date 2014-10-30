function trsv_unn_sym()
  condA = [];
  err_d = [];
  err_k = [];
  
  j=1;
  %unn
  alpha=50;
  for i = 1:21
    %[A, b] = trsv_gen_unn(alpha + 2*i, i);
    [A, b] = trsv_gen_unn_my(i);
    
    [condA(j), err_d(j), err_k(j)] = trsv_unn_exact(i, A, b);
    j = j+1;
    i
  end

  err_d
  err_k
  condA

  %ax = plotyy(condA, err_d, condA, err_k);
  loglog(condA, err_d, condA, err_k);
  %loglog(condA, err_k);
  %ax = plotyy(condA, err_d, condA, err_k, 'loglog');
  xlim([1, 10^50]);
  ylim([10^(-40), 10]);
  xlabel('CondA');
  ylabel('Error');
  legend('err_d','err_k');
  grid on;
  %ylabel(ax(2), 'Error Kulisch');
end

function [A, b] = trsv_gen_unn_my(n)
    A = triu(rand(n));
    for i = 1:n
        A(i,i) = 10^((-1)^i*10*(i-1)/n);
    end  
    b = rand(n, 1);
end

function x = trsv_unn_d(n, A, b)
  %x = A \ b;
  
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
  %trsv for unn matrices
  for i = n:-1:1
    s = sym(b(i), 'd');
    for j = i+1:n
      s = s - sym(A(i,j), 'd') * sym(x(j), 'd');
    end
    x(i) = double(s) / A(i, i);
  end
end

function [condA, err_d, err_k] = trsv_unn_exact(n, A, b)
  %double
  x_k = sym(trsv_unn_kulisch(n, A, b), 'd');
  x_d = sym(trsv_unn_d(n, A, b), 'd');

  A = sym(A, 'd');
  b = sym(b, 'd');

  %x_e = A \ b;
  %trsv for unn matrices
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
  A_inv = inv(A);
  %condA = cond(double(A), Inf);
  condA = norminf_m(A, n) * norminf_m(A_inv, n);
end

function res = norminf_m(A, n)
  res = 0.0;

  for i = 1:n
      sum = sym(0.0);
      for j = 1:n
          sum = sum + abs(A(i,j));
      end
      sum_d = double(abs(sum));
      if res < sum_d
          res = sum_d;
      end
  end
end

function [A, b] = trsv_gen_unn(alpha, n)
    A = zeros(n,n);
    
    for i = 1:n
        A(i,i) = 1;
    end
    for i = n-1:-2:1
        A(i,n) = 1;
    end    
    for i = n-2:-2:1
        for j = i+1:n
            A(i,j) = (-1)^(i+j+1) * 2^alpha;
        end
    end
    for i = n-3:-2:1
        for j = i+1:n-1
            A(i,j) = (-1)^(i+j);
        end
    end
    
    b = zeros(n, 1);
    for i = n:-2:1
        b(i) = 1;
    end
    for i = n-1:-2:1
        b(i) = 2;
    end    
end

