function [] = shrink(dat, gr, iter)
  
  load(strcat('data/1m/', int2str(dat), 'group0.mat'));
  
  if gr == 1
    train_vec = [cell2mat(group(1,1));cell2mat(group(1,2))];
  else
    train_vec = [cell2mat(group(2,1));cell2mat(group(2,2))];
  end
  
  uniQ = sort(unique(train_vec(:,2)));
  all = train_vec(:, 2);
  for i = 1:size(uniQ)
      all( all == uniQ(i) ) = i;
  end
  train_vec(:, 2) = all;
  
  uniQ = sort(unique(train_vec(:,1)));
  all = train_vec(:, 1);
  for i = 1:size(uniQ)
      all( all == uniQ(i) ) = i;
  end
  train_vec(:, 1) = all;
  
  [gr_u, gr_size, gr_t_size, gr_v_size, group] = groupping(max(train_vec(:,1)), 2, [0.5,0.5], train_vec);
  save(strcat('data/1m/', int2str(dat), 'group', int2str(gr), '.mat'), 'gr_u', 'gr_size', 'gr_t_size', 'gr_v_size', 'group');
  
end