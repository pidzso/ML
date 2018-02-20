function [mal, new_s, new_t_s] = manipulate(mal_gr_n, gr_size, gr_t_size, ...
          gr_v_size, group, gr_u, type, priv, sens, bound, fake)
  switch type 
  case 'hid'
    mal_gr = cell2mat(group(mal_gr_n, 1));
    r_perm = randperm(gr_t_size(mal_gr_n));
    mal_gr = mal_gr(r_perm,:);
    
    mal_gr(1:floor(priv * gr_t_size(mal_gr_n)), :) = [];
    gr_size(mal_gr_n)   = size(mal_gr, 1) + gr_v_size(mal_gr_n);
    gr_t_size(mal_gr_n) = size(mal_gr, 1);
    
    group(mal_gr_n, 1) = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    mal                = group;
    new_s              = gr_size;
    new_t_s            = gr_t_size;
  
  case 'ran'
    mal_gr  = cell2mat(group(mal_gr_n, 1));
    r_perm1 = randperm(size(mal_gr, 1));
    mal_gr  = mal_gr(r_perm1, :);
    r_perm2 = randperm(size(fake, 1));
    fake    = fake(r_perm2, :);
    
    mal_gr(1:floor(priv * size(mal_gr, 1)), :) = fake(1:floor(priv * size(mal_gr, 1)), :);
    
    group(mal_gr_n, 1) = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    mal                = group;
    new_s              = gr_size;
    new_t_s            = gr_t_size;
    
  case 'add'
    r_perm = randperm(size(fake, 1));
    fake   = fake(r_perm, :);
    mal_gr = [cell2mat(group(mal_gr_n, 1)); fake(1:floor(priv * gr_t_size(mal_gr_n)), :)];
    r_perm = randperm(size(mal_gr, 1));
    mal_gr = mal_gr(r_perm, :);

    % adding
    group(mal_gr_n, 1)  = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    gr_t_size(mal_gr_n) = size(mal_gr, 1);
    gr_size(mal_gr_n)   = gr_t_size(mal_gr_n) + gr_v_size(mal_gr_n);
   
    mal     = group;
    new_s   = gr_size;
    new_t_s = gr_t_size;
    
  case 'bdp'
    mal_gr = cell2mat(group(mal_gr_n, 1));
    rat    = mal_gr(:, 3);
    
    aux = rand(size(mal_gr, 1), 1) - 0.5;
    lap = sens / (priv * sqrt(2)) * sign(aux).* log(1 - 2 * abs(aux));
    
    rat = rat + lap;
    rat = bounding(rat, bound);
    mal_gr(:, 3) = rat;
    
    group(mal_gr_n, 1) = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    mal                = group;
    new_s              = gr_size;
    new_t_s            = gr_t_size;
    
  case 'udp'
    mal_gr   = cell2mat(group(mal_gr_n, 1));
    mal_v_gr = cell2mat(group(mal_gr_n, 2));
    
    oth_gr   = cell2mat(group(3-mal_gr_n, 1));
    oth_v_gr = cell2mat(group(3-mal_gr_n, 2));
    
    org = size(mal_gr, 1);
    
    % create rating matrix
    mx = zeros(size(union(union(mal_gr(:, 1), mal_v_gr(:, 1)), ...
                          union(oth_gr(:, 1), oth_v_gr(:, 1))), 1), ...
               size(union(union(mal_gr(:, 2), mal_v_gr(:, 2)), ...
                          union(oth_gr(:, 2), oth_v_gr(:, 2))), 1));
    % set ratings
    for i=1:size(mal_gr, 1)
      mx(mal_gr(i, 1), mal_gr(i, 2)) = mal_gr(i, 3);
    end
    
    % add noise
    aux = rand(size(mx)) - 0.5;
    lap = sens / (priv * sqrt(2)) * sign(aux).* log(1 - 2 * abs(aux));
    mx = mx + lap;
    
    % bound ratings
    mx = bounding(mx, bound);
    
    % remove verify set
    for i=1:size(mal_v_gr, 1)
      mx(mal_v_gr(i, 1), mal_v_gr(i, 2)) = 0;
    end
    
    % remove other group's set
    mx(oth_gr(:, 1), :)   = 0;
    mx(oth_v_gr(:, 1), :) = 0;
    
    % clamping 
    mx(abs(mx) < 0.5) = 0;
    aux = rand(size(mx(abs(mx) == 2), 1), 1) < 0.1;
    mx(abs(mx) == 2) = aux .* mx(abs(mx) == 2);
    
    [u, i, r] = find(mx);
    mal_gr    = [u, i, r];
    
%    fprintf(1, 'Size Change\t%f\n', size(mal_gr, 1) / org);
    
    group(mal_gr_n, 1)  = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    gr_size(mal_gr_n)   = size(mal_gr, 1) + gr_v_size(mal_gr_n);
    gr_t_size(mal_gr_n) = size(mal_gr, 1);
    
    mal     = group;
    new_s   = gr_size;
    new_t_s = gr_t_size;
  end
end