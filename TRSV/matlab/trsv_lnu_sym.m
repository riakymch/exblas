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

function x = trsv_lnu_d(n, A, b)
  %trsv for lnu matrices
  %x = A \ b;   
  for i = 1:n
    s = b(i);
    for j = 1:i-1
      s = s - A(i,j) * x(j);
    end
    x(i) = s / A(i, i);
  end
end

function x = trsv_lnu_kulisch(n, A, b)
  %trsv for lnu matrices
  for i = 1:n
    s = sym(b(i));
    for j = 1:i-1
        s = s - sym(A(i,j)) * sym(x(j));
    end
    x(i) = double(s) / A(i, i);
  end
end

function [condA, err_d, err_k] = trsv_lnu_exact(n, A, b)
  %double
  x_k = trsv_lnu_kulisch(n, A, b);
  x_d = trsv_lnu_d(n, A, b);

  A = sym(A, 'd');
  b = sym(b, 'd');
  
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
