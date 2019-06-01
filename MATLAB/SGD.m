function [ch1, ch2, act] = SGD(f_item, f_user, n_features, n_item, n_user, ...
             pre_rmse, rmse, n_group, group, gr_t_size, mal_type, priv, ...
             join, max_iter, bound_f, bound_rat, lambda, epsilon)
  iter = 0;
  while iter < max_iter
    iter = iter + 1;
    
    % training alone/joined
    if strcmp(join, 'all')
      actual_perm = randperm(n_group);
    else
      actual_perm = [str2double(join)];
    end
    
    for g = actual_perm;
      % randomizing order
      actual_g = cell2mat(group(g, 1));
      actual_g = actual_g(randperm(gr_t_size(g)), :);
%      fprintf(1, 'iter: %3i\t group: %3i\n', iter, g);
      
%      f_prev = f_item;
      % update feature matrix
      [f_item, f_user] = improve(f_item, f_user, n_features, max_iter, ...
                         bound_f, bound_rat, priv, n_item, n_user, ...
                         lambda, epsilon, actual_g, gr_t_size, g, mal_type);
      
      % bounding
      f_item = bounding(f_item, bound_f);
      f_user = bounding(f_user, bound_f);
                             
      % compute groupwise error
      [pre_rmse, rmse] = cal_rmse(n_group, group,  f_item, f_user, ...
                         bound_rat, lambda, pre_rmse, iter, rmse, g);
    end
  end
      
%  fprintf(1, 'Final RMSE\t%1.8f\t%1.8f\n', pre_rmse);
  act = pre_rmse; 
 
  % accumulating rmse change
  ch1 = squeeze(rmse(1:end,1,:));
  ch2 = squeeze(rmse(1:end,2,:));
  
%  fprintf(1, 'G1\t%1.8f\t%1.8f\n', sum(ch1(:,1)), sum(ch1(:,2)));
%  fprintf(1, 'G2\t%1.8f\t%1.8f\n', sum(ch2(:,1)), sum(ch2(:,2)));
end