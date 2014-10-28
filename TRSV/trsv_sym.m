function trsv_sym()
  condA = [];
  err_d = [];
  err_k = [];

  for i = 10:10:60
    A = triu(rand(i));
    alpha = i * i;
    for j=1:i
        A(j,j) = 1 - (1 - alpha^(-1)) * (j - 1) / (i - 1);
        %A(j,j) = 1.0;
    end
    %A(i/2,i/2) = alpha^(-1);

    b = rand(i, 1);

    [condA(i/10), err_d(i/10), err_k(i/10)] = trsv_unn_exact(i, A, b);
    i
  end

  err_d
  err_k
  condA

  %ax = plotyy(condA, err_d, condA, err_k);
  plot(condA, err_d, condA, err_k);
  xlabel('CondA');
  ylabel('Error Double');
  legend('err_d','err_k')
  %ylabel(ax(2), 'Error Kulisch');
end

function x = trsv_unn_d(n, A, b)
  x = A \ b;
end

function x = trsv_unn_kulisch(n, A, b)
  x = zeros(n, 1);

  %trsv for unn matrices
  for i = n:-1:1
    s = sym(b(i));
    for j = i+1:n
      s = s - sym(A(i,j)) * sym(x(j));
    end
    x(i) = double(s) / A(i, i);
  end
end

function [condA, err_d, err_k] = trsv_unn_exact(n, A, b)
  %double
  x_k = sym(trsv_unn_kulisch(n, A, b));
  x_d = sym(trsv_unn_d(n, A, b));

  A = sym(A);
  b = sym(b);

  x_e = A \ b;

  %compute error
  norm_x_e = max(double(abs(x_e)));
  err_k = max(double(abs(x_e - x_k))) / norm_x_e;
  err_d = max(double(abs(x_e - x_d))) / norm_x_e;

  %compute cond number
  A_inv = inv(A);
  condA = norminf_m(A, n) * norminf_m(A_inv, n);
end

function res = norminf_m(A, n)
  res = 0.0;

  for i = 1:n
      sum = sym(0.0);
      for j = 1:n
          if i >= j
            sum = sum + abs(A(i,j));
          end
      end
      sum_d = double(abs(sum));
      if res < sum_d
          res = sum_d;
      end
  end
end

