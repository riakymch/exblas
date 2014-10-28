function trsv_sym()
  condA = [];
  condA_kulisch = [];
  err = [];
  err_kulisch = [];
  
  for i = 1:10
    [condA(i), err(i)] = trsv_unn_kulisch(i * 10);
    i
  end

  err
  condA

  %figure(1, "visible", "off");
  ax = plot(condA, err);
  xlabel('CondA');
  ylabel('Error');
end

function x = trsv_unn(n)
  A = triu(rand(n));
  b = rand(n, 1);
  x = rand(n, 1);
  
  %trsv for unn matrices
  for i = n:-1:1
    s = b(i);
    for j = i+1:n
      s = s - A(i,j) * x(j);
    end
    x(i) = s / A(i, i);
  end
  %x = A \ b;
  
  %verify
  %err = norm(A * x - b, Inf);
  %condA = cond(A, Inf);
end

function [condA, err] = trsv_unn_kulisch(n)
  A = sym(triu(rand(n)));
  b = sym(rand(n, 1));
  x = sym(rand(n, 1)); 
 
  %trsv for unn matrices
  for i = n:-1:1
    s = b(i);
    for j = i+1:n
      s = s - A(i,j) * x(j);
    end
    x(i) = s / A(i, i);
  end
  
  %compute error
  %err_v = A * x - b;
  %err = max(double(abs(err_v)));
  x_comp = sym(trsv_unn(n));
  err = max(double(abs(x - x_comp))) / max(double(abs(x)));
  
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
