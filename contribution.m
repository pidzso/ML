function metric = contribution(n_group, n_item, f_item, f_prev, iter, g)
  metric_loc = zeros(n_item, n_group);
  for j=1:n_item
    % last step
    norm_O = double(norm(f_item(j, :) - f_prev(j, :)));  
    norm_R = double(norm(f_prev(j, :))); 
    norm_A = double(norm(f_item(j, :)));

    metric_loc(j, g) = (norm_A^2 + norm_O^2 - norm_R^2) / (2 * norm_A^2);
      
    % remove NaN
    ix = isnan(metric_loc(:, g));
    metric_loc(find(ix), g) = 0;
  end
  metric = sum(metric_loc(:, g)) / n_item;
end