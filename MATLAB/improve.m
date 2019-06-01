function [new_i, new_u] = improve(f_item, f_user, n_features, max_iter, ...
                          bound_f, bound_rat, priv, n_item, n_user, ...
                          lambda, epsilon, actual_g, gr_t_size, g, mal_type)
      
  actual_u = double(actual_g(:, 1));
  actual_i = double(actual_g(:, 2));
  actual_r = double(actual_g(:, 3));

    % compute predictions
    pred  = sum(f_item(actual_i, :) .* f_user(actual_u, :), 2);
    pred = bounding(pred, bound_rat);
    
    % compute pre gradients
    err   = repmat(pred - actual_r, 1, n_features);
    err = bounding(err, 2 * bound_rat);
    reg_i = err .* f_user(actual_u, :) + lambda * f_item(actual_i, :);
    reg_u = err .* f_item(actual_i, :) + lambda * f_user(actual_u, :);

    grad_i = zeros(n_item, n_features);
    grad_u = zeros(n_user, n_features);

    for j=1:gr_t_size(g)
      grad_i(actual_i(j), :) = grad_i(actual_i(j), :) +  reg_i(j, :);
      grad_u(actual_u(j), :) = grad_u(actual_u(j), :) +  reg_u(j, :);
    end
    
    sens = epsilon * (2 * bound_rat * bound_f + lambda * bound_f);
    grad_i = noise(n_item, n_features, grad_i, max_iter, ...
                   epsilon, sens, mal_type, priv, g);
    
    % update features
    new_u = f_user - epsilon * grad_u;
    new_i = f_item - grad_i;