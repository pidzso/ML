function [mal, new_s, new_t_s] = manipulate( ...
         mal_gr_n, gr_size, gr_t_size, gr_v_size, group, type, priv, fake)
switch type
  case 'hid'
    mal_gr = cell2mat(group(mal_gr_n, 1));
    r_perm = randperm(gr_t_size(mal_gr_n));
    mal_gr = mal_gr(r_perm,:);
    
    mal_gr(1:floor(priv * gr_t_size(mal_gr_n)),:) = [];
    gr_size(mal_gr_n)   = size(mal_gr, 1) + gr_v_size(mal_gr_n);
    gr_t_size(mal_gr_n) = size(mal_gr, 1);
    
    group(mal_gr_n, 1) = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    mal                = group;
    new_s              = gr_size;
    new_t_s            = gr_t_size;
  
  case 'ran'
    mal_gr  = cell2mat(group(mal_gr_n, 1));
    r_perm1 = randperm(size(mal_gr, 1));
    mal_gr  = mal_gr(r_perm1,:);
    r_perm2 = randperm(size(fake, 1));
    fake    = fake(r_perm2, :);
    
    mal_gr(1:floor(priv * size(mal_gr, 1)), 3) = fake(1:floor(priv * size(mal_gr, 1)), 3);
    
    group(mal_gr_n, 1) = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    mal                = group;
    new_s              = gr_size;
    new_t_s            = gr_t_size;
    
  case 'add'
    r_perm = randperm(size(fake, 1));
    fake   = fake(r_perm, :);
    mal_gr = [cell2mat(group(mal_gr_n, 1)); fake(1:floor(priv * gr_t_size(mal_gr_n)), :)];

    % adding
    group(mal_gr_n, 1)  = mat2cell(mal_gr, size(mal_gr, 1), size(mal_gr, 2));
    gr_t_size(mal_gr_n) = size(mal_gr, 1);
    gr_size(mal_gr_n)   = gr_t_size(mal_gr_n) + gr_v_size(mal_gr_n);
   
    mal     = group;
    new_s   = gr_size;
    new_t_s = gr_t_size;
    
end