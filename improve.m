function [new_i, new_u] = improve( ...
         f_item, f_user, n_features, n_item, n_user, lambda, epsilon, actual_g, gr_t_size, g)
  actual_u = double(actual_g(:, 1));
  actual_i = double(actual_g(:, 2));
  actual_r = double(actual_g(:, 3));

    % compute predictions
    pred  = sum(f_item(actual_i, :) .* f_user(actual_u, :), 2);

    % compute gradients
    aux   = repmat(2 * (pred - actual_r), 1, n_features);
    reg_i = aux .* f_user(actual_u, :) + lambda * f_item(actual_i, :);
    reg_u = aux .* f_item(actual_i, :) + lambda * f_user(actual_u, :);
    clear aux;

    grad_i = zeros(n_item, n_features);
    grad_u = zeros(n_user, n_features);

    for j=1:gr_t_size(g)
      grad_i(actual_i(j), :) = grad_i(actual_i(j), :) +  reg_i(j, :);
      grad_u(actual_u(j), :) = grad_u(actual_u(j), :) +  reg_u(j, :);
    end

    clear reg_i;
    clear reg_u;

    % update features
    new_u = f_user - epsilon * grad_u / gr_t_size(g);
    new_i = f_item - epsilon * grad_i / gr_t_size(g);
    
    clear grad_u;
    clear grad_i;