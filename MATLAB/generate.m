function out = generate(n_item, n_group, gr_u, gr_t_size, group)
  gen = 2; % depending on the density of the dataset
  out = cell(1, 2);
  for g=1:n_group
    n_gr_u = gr_u(:, g); 
    n_gr_u(n_gr_u == 0) = [];
    fake   = zeros(gr_t_size(g) * gen, 2);
    gr_t   = cell2mat(group(g, 1));
    gr_v   = cell2mat(group(g, 2));
    gr_all = [gr_t(:, 1), gr_t(:, 2); gr_v(:, 1), gr_v(:, 2)];
    
    % chosing user-item pair
    for j=1:gr_t_size(g) * gen
      aux_u      = n_gr_u(randi(size(n_gr_u, 1)));
      aux_i      = randi(n_item);
      fake(j, :) = [aux_u, aux_i];
    end
    
    % removing
    fake       = unique(fake, 'rows');
    duplicate = intersect(fake, gr_all, 'rows');
    fake      = setxor(fake, duplicate, 'rows');
    
    % generate ratings
    rat  = [gr_t(:, 3); gr_v(:, 3)];
    fake = [fake, randn(size(fake, 1), 1)];
    
    r_perm = randperm(size(fake, 1));
    fake   = fake(r_perm, :);
    fake   = fake(1:gr_t_size(g), :);
    
    out(1, g) = mat2cell(fake, gr_t_size(g), 3);
  end
end
    