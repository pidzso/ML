function new_grd = noise(n_item, n_features, grad_i, ...
                         max_iter, epsilon, sens, mal_type, priv, g)
  init = 0; % TODO
  if and(strcmp(mal_type(g,:), 'pgd'), priv(g) > 0)
    % http://nl.mathworks.com/matlabcentral/fileexchange/13705-laplacian-random-number-generator
    aux   = rand(n_item, n_features) - 0.5;
    sigma = n_features * n_item * (max_iter - init) * sens / priv(g);
    lap   = sigma / sqrt(2) * sign(aux) .* log(1 - 2 * abs(aux));
    new_grd = epsilon * grad_i + lap;
  else
    new_grd = epsilon * grad_i;
  end
    %new_grd(new_grd > 0)  = 0.1;
    %new_grd(new_grd <= 0) = -0.1;
    new_grd = bounding(new_grd, 0.1); % TODO
end