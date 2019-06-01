function [mal, new_s, new_t_s] = manipulate(mal_gr_n, gr_size, gr_t_size, ...
          gr_v_size, group, type, priv, sens, bound)
  
  mal     = 0;
  new_s   = 0;
  new_t_s = 0;    
      
  switch type 
  
  case 'sup'
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
  end
end