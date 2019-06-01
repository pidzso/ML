function [pr_d, pr_i, pr_u, pr_r, avg_u, avg_i] = preproc(data, limit)
  pr_d  = data; 
  new_u = data(:, 1);
  new_i = data(:, 2);
  new_r = data(:, 3);
  [user_rat, users] = hist( new_u, unique(new_u));
  [item_rat, items] = hist( new_i, unique(new_i));
  
  % remove outliers  
  ind = 0;
  while ind == 0
  
    fprintf(1, 'filtering\n');
    
    aux = size(pr_d);
    
    for i=1:size(item_rat, 2)
      if item_rat(i) < limit
        new_i(new_i == items(i)) = 0;
      end
    end
    for i=1:size(new_i)
      if new_i(i) == 0
        new_u(i) = 0;
        new_r(i) = 0;
      end
    end
    
    for u=1:size(user_rat, 2)
      if user_rat(u) < limit
        new_u(new_u == users(u)) = 0;
      end
    end
    for u=1:size(new_u)
      if new_u(u) == 0
        new_i(u) = 0;
        new_r(u) = 0;
      end
    end
    
    new_i(new_i == 0) = [];
    new_u(new_u == 0) = [];
    new_r(new_r == 0) = [];
    pr_d = [new_u, new_i, new_r];
    if aux == size(pr_d)
        ind = 1;
    end
  end
  
  fprintf(1, 'recounting users\n');
  
  uniQ = sort(unique(pr_d(:, 1)));
  all   = pr_d(:, 1);
  for u = 1:size(uniQ)
    all( all == uniQ(u) ) = u;
  end
  pr_d(:, 1) = all;
  
  fprintf(1, 'recounting items\n');
  
  uniQ = sort(unique(pr_d(:, 2)));
  all   = pr_d(:, 2);
  for i = 1:size(uniQ)
    all( all == uniQ(i) ) = i;
  end
  pr_d(:, 2) = all;
  
  pr_u = max(pr_d(:, 1));
  pr_i = max(pr_d(:, 2));
  pr_r = size(pr_d, 1);
  
  [n_user_rat, n_users] = hist(new_u, unique(new_u));
  [n_item_rat, n_items] = hist(new_i, unique(new_i));
  
  % calculate item/user averages
  avg_i = n_items;
  avg_u = n_users;
  
  fprintf(1, 'averaging items\n');
  
  for i=1:pr_i
    aux = pr_d(pr_d(:, 2) == i, 3);
    avg_i(i) = sum(aux) / n_item_rat(i);
    pr_d(pr_d(:, 2) == i, 3) = aux - avg_i(i);
  end
  
  fprintf(1, 'averaging users\n');
  
  for u=1:pr_u
    aux = pr_d(pr_d(:, 1) == u, 3);
    avg_u(u) = sum(aux) / n_user_rat(u);
    pr_d(pr_d(:, 1) == u, 3) = aux - avg_u(u);
  end 
end