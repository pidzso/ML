function [post_rmse, new_r] = ...
         cal_rmse(n_group, group, f_item, f_user, lambda, pre_rmse, iter, rmse, varargin)
  for j=1:n_group
    % selecting group
    ver   = cell2mat(group(j, 2));
    ver_u = double(ver(:, 1));
    ver_i = double(ver(:, 2));
    ver_r = double(ver(:, 3));
    ver_p = sum(f_item(ver_i, :) .* f_user(ver_u, :), 2);
    error = sum((ver_p - ver_r).^2 + 0.5 * lambda * ...
            (sum((f_item(ver_i, :).^2 + f_user(ver_u, :).^2), 2)));
     
    % selecting round
    if iter == 0
      pre_rmse(j) = sqrt(error / size(ver, 1));
      fprintf(1, '%1.8f\t', pre_rmse(j));
    else
      g                = varargin{1};
      rmse(iter, g, j) = pre_rmse(j) - sqrt(error / size(ver, 1));
      pre_rmse(j)      = sqrt(error / size(ver, 1));
    end
    post_rmse = pre_rmse;
    new_r     = rmse;
  end