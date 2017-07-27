function SGD(priv, mal_type, dat)
  clearvars -except priv mal_gr_n mal_type dat
  fprintf(1, 'Dat/Pri/Typ\t%s/%1.2f/%1.2f/%s/%s\n', ...
          dat, priv, mal_type);
  
  % parameters
  lambda     = 0.01; % regularization
  iter       = 0;    % iterator
  max_iter   = 200;  % iteration
  n_features = 10;   % number of features
  n_group    = 2;    % number of groups
  gr_div     = [1/2, 1/2]; % division of groups
  
  % learning rate
  switch dat
    case '100k'
      epsilon = 6.25;
    case '1m'
      epsilon = 25;
    case '10m'
      epsilon = 100;
    case '20m'
      epsilon = 150;
  end
  
  % stopping parameters
  s_type     = ['siz'; 'siz'; 'siz'; 'dir'; 'dir'; 'dir'];
  toleration = [0, 0, 2, 5];
  buffer     = [2, 5, 2, 5];
  stopped    = [0, 0, 0, 0; 0, 0, 0, 0];
  
  data = strcat('data/', dat, '/', dat, '.mat');
  load(data);
  
  % matrices
  f_item = 0.1 * randn(n_item, n_features);   % movie feature vectors
  f_user = 0.1 * randn(n_user, n_features);   % user feature vecators
  rmse   = zeros(max_iter, n_group, n_group); % group error
  
  % new metric variables
  %f_prev     = f_item;                   % before update
  %usefulness = zeros(max_iter, n_group); % value of iteration
  
  % groupping
  %[gr_u, gr_size, gr_t_size, gr_v_size, group] = ...
  %          groupping(n_user, n_group, gr_div, train_vec);
  %data = strcat('data/', dat, '/g', dat(1:end-1), '.mat');
  %load(data);
  load('data/1m/g1.mat');
  
  % generate fake ratings
  fake = generate(n_item, n_group, gr_u, gr_t_size, group);
  
  % compute error
  %fprintf(1, 'Starting RMSE\n');
  [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
                              lambda, zeros(n_group, 1), iter, rmse, []);
  
  % manipulate group
  for g=1:n_group
    in = cell2mat(fake(1,g));
    if priv(g) > 0
      [group, gr_size, gr_t_size] = manipulate( ...
       g, gr_size, gr_t_size, gr_v_size, group, mal_type(g,:), priv(g), in);
    end
  end
  
  while iter < max_iter
    iter        = iter + 1;
    actual_perm = randperm(n_group);
    
    for g = actual_perm;
      % randomizing order
      actual_g = cell2mat(group(g, 1));
      actual_g = actual_g(randperm(gr_t_size(g)), :);
      %fprintf(1, 'iter: %3i\t group: %3i\n', iter, g);
      
      %f_prev = f_item;
      % update feature matrix
      [f_item, f_user] = improve(f_item, f_user, n_features, n_item, ...
                                 n_user, lambda, epsilon, actual_g, gr_t_size, g);
      
      % compute groupwise error
      [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
                                  lambda, pre_rmse, iter, rmse, g);
      
      % compute new measure
      %usefulness(iter, g) = contribution(n_group, n_item, f_item, f_prev, iter, g);
      
      % stopping condition
      for s=1:4
        if stopped(g, s) == 0 && stop(s_type(s,:), toleration(s), iter, buffer(s), rmse, g) == 1
          stopped(g, s) = iter;
        end 
      end
    end
  end
  
  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i', stopped(1,:));
  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i\n', stopped(2,:));
      
  fprintf(1, 'Final RMSE\n%1.8f\t%1.8f\n', pre_rmse);
  
  % hadling rmse
  init = 1; % skipping the first x iteration
  ch1 = squeeze(rmse(init:end,1,:));
  ch2 = squeeze(rmse(init:end,2,:));
  
  %fprintf(1, 'Metric:\t%2.6f\t%2.6f\n', sum(usefulness));
  fprintf(1, 'G1 effect\n%1.8f\t%1.8f\n', sum(ch1(:,1)), sum(ch1(:,2)));
  fprintf(1, 'G2 effect\n%1.8f\t%1.8f\n', sum(ch2(:,1)), sum(ch2(:,2)));
end