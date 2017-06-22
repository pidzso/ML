function [fake1, fake2] = generate(n_item, n_group, gr_u, gr_t_size, group)
  for g=1:n_group
    gen    = 2; % depending on the density of the dataset
    fake   = zeros(gr_t_size(g) * gen, 2);
    gr_t   = cell2mat(group(g, 1));
    gr_v   = cell2mat(group(g, 2));
    gr_all = [gr_t(:,1), gr_t(:,2); gr_v(:,1), gr_v(:,2)];
    
    % chosing user-item pair
    for j=1:gr_t_size(g) * gen
      aux_u      = gr_u(randi(size(gr_u, 1)), g);
      aux_i      = randi(n_item);
      fake(j, :) = [aux_u, aux_i];
    end
    
    % removing
    fake       = unique(fake, 'rows');
    duplicate = intersect(fake, gr_all, 'rows');
    fake      = setxor(fake, duplicate, 'rows');
    
    % generate ratings
    rat    = [gr_t(:, 3); gr_v(:, 3)];
    rat_ty = unique(rat);
    aux    = zeros(size(rat_ty, 1), 1);
    
    for i=1:size(rat_ty, 1)
      aux(i) = size(rat(rat == rat_ty(i)), 1);
    end
    
    aux = aux / size(rat, 1);
    pd  = makedist('Multinomial', 'probabilities', aux);
    
    % 1m:    1- 2- 3- 4- 5
    % 10m20m:1-1.5...4.5-5
    if size(rat_ty, 1) > 5
      fake = [fake, random(pd , size(fake, 1), 1) / 2];
    else
      fake = [fake, random(pd , size(fake, 1), 1)];
    end
    
    r_perm = randperm(size(fake, 1));
    fake   = fake(r_perm, :);
    fake   = fake(1:gr_t_size(g), :);
    
    if g == 1
      fake1 = fake;
    else
      fake2 = fake;
    end
  end
    