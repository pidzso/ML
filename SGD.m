function SGD(priv, mal_type, gr_div, dat, join)
  clearvars -except priv mal_type gr_div dat join
%  fprintf(1, 'Dat/Pri/Typ\t%s/%1.2f/%1.2f/%s/%s\n', ...
%          dat, priv, mal_type(1,:), mal_type(2,:));
  
  % parameters
  lambda     = 0.01; % regularization
  iter       = 0;    % iterator
  max_iter   = 20;   % iteration
  n_features = 5;    % number of features
  bound_f    = 0.5;  % feature bound
  bound_avg  = 2;    % user average bound
  bound_err  = 4;    % error bound
%  bound_rat  = 1;    % rating bound
  n_group    = 2;    % number of groups
  rmse       = zeros(max_iter, n_group, n_group); % group error
  
  % learning rate
  switch dat
    case '100k'
      epsilon = 25;
    case '1m'
      epsilon = 6/1000;
    case '10m'
      epsilon = 300;
    case '20m'
      epsilon = 600;
    case 'netflix'
      epsilon = 400;
  end
  
  % stopping parameters
%  s_type     = ['siz'; 'siz'; 'siz'; 'dir'; 'dir'; 'dir'];
%  toleration = [0, 0, 2, 5];
%  buffer     = [2, 5, 2, 5];
%  stopped    = [0, 0, 0, 0; 0, 0, 0, 0];
  
  % read raw input
%  data = strcat('data/', dat, '/', dat, '.mat');
%  load(data);
%  clear data;
  
  % preprocess data
  data = strcat('data/', dat, '/proc_', dat, '.mat');
%  [proc_d, proc_i, proc_u, proc_r, avg_u, avg_i] = preproc(train_vec);
%  avg_u = bounding(avg_u, bound_avg);
%  proc_d(:, 3) = bounding(proc_d(:, 3), bound_rat); % bullshit
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
%  [gr_u, gr_size, gr_t_size, gr_v_size, group] = ...
%            groupping(proc_u, n_group, gr_div, proc_d);
%  save(data, 'gr_u', 'gr_size', 'gr_t_size', 'gr_v_size', 'group');
  load(data);
  clear data;
  
  % compute error
  [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
   lambda, zeros(n_group, 1), iter, rmse, []);
%  fprintf(1, 'Starting RMSE\n%1.8f\t%1.8f\n', pre_rmse(1), pre_rmse(2));
  
  % manipulate group
%  if mal_type ~= 'pgd'
%   fake = generate(n_item, n_group, gr_u, gr_t_size, group);
%    for g=1:n_group
%      in = cell2mat(fake(1, g));
%      if priv(g) > 0
%        [group, gr_size, gr_t_size] = manipulate( ...
%         g, gr_size, gr_t_size, gr_v_size, group, mal_type(g,:), priv(g), in);
%      end
%    end
%  end
  
  while iter < max_iter
    iter        = iter + 1;
    
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
                         bound_f, bound_err, priv, proc_i, proc_u, ...
                         lambda, epsilon, actual_g, gr_t_size, g);
      
      % bounding
      bounding(f_item, bound_f);
      bounding(f_user, bound_f);
                             
      % compute groupwise error
      [pre_rmse, rmse] = cal_rmse(n_group, group,  f_item, f_user, ...
       lambda, pre_rmse, iter, rmse, g);
      
      % compute new measure
%      usefulness(iter, g) = contribution(n_group, n_item, f_item, f_prev, iter, g);
      
      % stopping condition
%      for s=1:4
%       if stopped(g, s) == 0 && stop(s_type(s,:), toleration(s), iter, buffer(s), rmse, g) == 1
%          stopped(g, s) = iter;
%        end 
%      end
    end
  end
  
%  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i', stopped(1,:));
%  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i', stopped(2,:));
      
%  fprintf(1, '\nFinal RMSE\n%1.8f\t%1.8f', pre_rmse);
  fprintf(1, '%1.8f\t%1.8f\n', pre_rmse);
  
  % accumulating rmse change
  init = 1; % skipping the first x iteration
  ch1 = squeeze(rmse(init:end,1,:));
  ch2 = squeeze(rmse(init:end,2,:));
  
%  fprintf(1, 'Metric:\t%2.6f\t%2.6f\n', sum(usefulness));
  fprintf(1, '\nG1 effect\n%1.8f\t%1.8f', sum(ch1(:,1)), sum(ch1(:,2)));
  fprintf(1, '\nG2 effect\n%1.8f\t%1.8f\n', sum(ch2(:,1)), sum(ch2(:,2)));
end