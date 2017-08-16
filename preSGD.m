function [g1, g2] = preSGD(priv, mal_type, gr_div, dat, join)
  clearvars -except priv mal_type gr_div dat join
  fprintf(1, 'Dat/Pri/Typ\t%s/%1.2f/%1.2f/%s/%s\n', ...
          dat, priv, mal_type(1,:), mal_type(2,:));
  
  [lambda, max_iter, n_features, n_group, epsilon, ...
          bound_f, bound_avg, bound_err, bound_rat, ...
          s_type, toleration, buffer, stopped] = param(dat);
  rmse = zeros(max_iter, n_group, n_group); % group error
  
  % read raw input
  data = strcat('data/', dat, '/', dat, '.mat');
  load(data);
  clear data;
  
  % preprocess data
  data = strcat('data/', dat, '/proc_', dat, '.mat');
%  [proc_d, proc_i, proc_u, proc_r, avg_u, avg_i] = preproc(train_vec);
%  avg_u = bounding(avg_u, bound_avg); % not needed
%  save(data, 'proc_d', 'proc_i', 'proc_u', 'proc_r', 'avg_u', 'avg_i');
  load(data);
  clear data;  
  
  % matrices
  f_item = 0.1 * randn(proc_i, n_features);   % movie feature vectors
  f_user = 0.1 * randn(proc_u, n_features);   % user feature vectors
  
  % bounding
  f_user = bounding(f_user, bound_f);
  f_item = bounding(f_item, bound_f);
  
  % new metric variables
%  f_prev     = f_item;                   % before update
%  usefulness = zeros(max_iter, n_group); % value of iteration
  
  % groupping
  data = strcat('data/', dat, '/', num2str(gr_div(1) * 10), ...
                '-', num2str(gr_div(2)*10), '.mat');
  [gr_u, gr_size, gr_t_size, gr_v_size, group] = ...
            groupping(proc_u, n_group, gr_div, proc_d);
%  save(data, 'gr_u', 'gr_size', 'gr_t_size', 'gr_v_size', 'group');
%  load(data);
  clear data;
  
  % compute error
  [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
   lambda, zeros(n_group, 1), 0, rmse, []);
  fprintf(1, 'Start RMSE\t%1.8f\t%1.8f\n', pre_rmse(1), pre_rmse(2));
  
  % manipulate group
  if sum(ismember(['hid'; 'ran'; 'add'; 'dif'], mal_type, 'rows')) > 0
    fake = generate(proc_i, n_group, gr_u, gr_t_size, group);
    for g=1:n_group
      in = cell2mat(fake(1, g));
      if priv(g) > 0
        [group, gr_size, gr_t_size] = manipulate(g, gr_size, gr_t_size, ...
         gr_v_size, group, mal_type(g,:), priv(g), in);
      end
    end
  end
  [g1, g2] = SGD(f_item, f_user, n_features, proc_i, proc_u, ...
      pre_rmse, rmse, n_group, group, gr_t_size, mal_type, priv, ...
      join, max_iter, bound_f, bound_err, lambda, epsilon);
end