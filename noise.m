function new_grd = noise(n_item, n_features, grad_i, max_iter, epsilon, sens, priv, g)
  if priv(g) > 0
    aux   = rand(n_item, n_features) - 0.5;
    lap   = epsilon * n_features * n_item * max_iter * sens / ...
            (priv(g) * sqrt(2)) * sign(aux).* log(1 - 2 * abs(aux));
    new_grd = grad_i + lap;
  else
    new_grd = grad_i;
  end
end