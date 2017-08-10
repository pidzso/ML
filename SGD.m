function [ch1, ch2] = SGD(f_item, f_user, n_features, n_item, n_user, ...
             pre_rmse, rmse, n_group, group, gr_t_size, mal_type, priv, ...
             join, max_iter, bound_f, bound_err, lambda, epsilon)
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
                         bound_f, bound_err, priv, n_item, n_user, ...
                         lambda, epsilon, actual_g, gr_t_size, g, mal_type);
      
      % bounding
      f_item = bounding(f_item, bound_f);
      f_user = bounding(f_user, bound_f);
                             
      % compute groupwise error
      [pre_rmse, rmse] = cal_rmse(n_group, group,  f_item, f_user, ...
       lambda, pre_rmse, iter, rmse, g);
      
      % compute new measure
%      usefulness(iter, g) = contribution(n_group, n_item, f_item, f_prev, iter, g);
      
      % stopping condition
%      for s=1:4
%       if and(stopped(g, s) == 0, ...
%          stop(s_type(s,:), toleration(s), iter, buffer(s), rmse, g) == 1)
%          stopped(g, s) = iter;
%        end 
%      end
    end
  end
  
%  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i', stopped(1,:));
%  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i', stopped(2,:));
      
  fprintf(1, 'Final RMSE\t%1.8f\t%1.8f', pre_rmse);
  
  % accumulating rmse change
  init = 1; % skipping the first x iteration
  ch1 = squeeze(rmse(init:end,1,:));
  ch2 = squeeze(rmse(init:end,2,:));
  
%  fprintf(1, 'Metric:\t%2.6f\t%2.6f\n', sum(usefulness));
  fprintf(1, '\nG1\t%1.8f\t%1.8f', sum(ch1(:,1)), sum(ch1(:,2)));
  fprintf(1, '\nG2\t%1.8f\t%1.8f\n', sum(ch2(:,1)), sum(ch2(:,2)));
end