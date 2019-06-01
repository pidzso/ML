function [g1, g2, act, init] = preSGD(priv, mal_type, gr_div, dat, join)
  
%  fprintf(1, '%s-%s-%s/1:%i/%1.2f-%1.2f/%s-%s\n', dat(1), ...
%          dat(2:end-1), dat(end), gr_div(2) / gr_div(1), ...
%          priv, mal_type(1,:), mal_type(2,:));
      
  [lambda, max_iter, n_features, n_group, epsilon, ...
          bound_f, bound_avg, bound_rat, ...
          s_type, toleration, buffer, stopped] = param(dat);
  rmse = zeros(max_iter, n_group, n_group); % group error
    
  load(strcat('data/1m/', int2str(dat/10), 'group', int2str(mod(dat,10)), '.mat'));
  x1 = cell2mat(group(1,1));
  x2 = cell2mat(group(1,2));
  x3 = cell2mat(group(2,1));
  x4 = cell2mat(group(2,2));
  proc_i = max([max(x1(:,2)),max(x2(:,2)),max(x3(:,2)),max(x4(:,2))]);
  proc_u = max([max(x1(:,1)),max(x2(:,1)),max(x3(:,1)),max(x4(:,1))]);

  % base recomendation
  f_item = zeros(proc_i, n_features);   % movie feature vectors
  f_user = zeros(proc_u, n_features);   % user feature vectors
  
  % initialize features
  f_item = 0.1 * randn(proc_i, n_features);   % movie feature vectors
  f_user = 0.1 * randn(proc_u, n_features);   % user feature vectors
  
  % bounding
  f_user = bounding(f_user, bound_f);
  f_item = bounding(f_item, bound_f);
  
  % compute error
  [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
   bound_rat, lambda, zeros(n_group, 1), 0, rmse, []);
  init = pre_rmse;
%  fprintf(1, 'Start RMSE\t%1.8f\t%1.8f\n', pre_rmse(1), pre_rmse(2));
  
  % manipulate group
  if sum(ismember(['sup'; 'bdp'], mal_type, 'rows')) > 0
    sens = max_iter * n_features * epsilon * ...
           (2 * bound_rat * bound_f + lambda * bound_f);
    for g=1:n_group
      if priv(g) > 0
        [group, gr_size, gr_t_size] = manipulate(g, gr_size, gr_t_size, ...
         gr_v_size, group, mal_type(g, :), priv(g), sens, bound_rat);
      end
    end
  end
  
  [g1, g2, act] = SGD(f_item, f_user, n_features, proc_i, proc_u, ...
      pre_rmse, rmse, n_group, group, gr_t_size, mal_type, priv, ...
      join, max_iter, bound_f, bound_rat, lambda, epsilon);
end