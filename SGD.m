function SGD(priv, mal_gr_n, mal_type, dat)
  clearvars -except priv mal_gr_n mal_type dat
  fprintf(1, 'Dat/Mal/Pri/Typ\t%s/%1i/%1.2f/%s\n', ...
          dat, mal_gr_n, priv, mal_type);
  
  % parameters
  epsilon    = 50;   % learning rate
  lambda     = 0.01; % regularization
  iter       = 0;    % iterator
  max_iter   = 200;  % iteration
  n_features = 10;   % number of features
  n_group    = 2;    % number of groups
  
  % stopping parameters
  s_type     = ['siz'; 'siz'; 'siz'; 'dir'; 'dir'; 'dir'];
  toleration = [0, 0, 0, 2, 3, 5];
  buffer     = [2, 3, 5, 2, 3, 5];
  stopped    = [max_iter, max_iter, max_iter, max_iter, max_iter, max_iter;...
                max_iter, max_iter, max_iter, max_iter, max_iter, max_iter;];
  why        = [0, 0, 0, 0, 0, 0; 0, 0, 0, 0, 0, 0];
  
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
  %          groupping(n_user, n_group, train_vec);
  data = strcat('data/', dat, '/g', dat(1:end-1), '.mat');
  load(data);
  
  % generate fake ratings
  [fake1, fake2] = generate(n_item, n_group, gr_u, gr_t_size, group);
  if mal_gr_n == 1
    fake = fake1;
  else
    fake = fake2;
  end
  
  % compute error
  fprintf(1, 'Starting RMSE\n');
  [pre_rmse, rmse] = cal_rmse(n_group, group, f_item, f_user, ...
                              lambda, zeros(n_group, 1), iter, rmse, []);
  
  % manipulate group
  if priv > 0
    [group, gr_size, gr_t_size] = manipulate( ...
      mal_gr_n, gr_size, gr_t_size, gr_v_size, group, mal_type, priv, fake);
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
      
      %stopping condition
      for s=1:6
        if why(g, s) == 0 && stop(s_type(s,:), toleration(s), iter, buffer(s), rmse, g) == 1
          stopped(g, s) = iter;
          why(g, s)     = g;
        end 
      end
    end
  end
  
  fprintf(1, '\niter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i\t\nwhy\t\t%1i\t%1i\t%1i\t%1i\t%1i\t%1i\n', ...
          stopped(1,:), why(1,:));
  fprintf(1, 'iter\t%3i\t%3i\t%3i\t%3i\t%3i\t%3i\t\nwhy\t\t%1i\t%1i\t%1i\t%1i\t%1i\t%1i\n', ...
          stopped(2,:), why(2,:));
      
  fprintf(1, 'Final RMSE\n%1.8f\t%1.8f\n', pre_rmse);
  
  % hadling rmse
  ch1 = squeeze(rmse(:,1,:));
  ch2 = squeeze(rmse(:,2,:));
  
  %fprintf(1, 'Metric:\t%2.6f\t%2.6f\n', sum(usefulness));
  fprintf(1, 'G1 effect\n%1.8f\t%1.8f\n', sum(ch1(:,1)), sum(ch1(:,2)));
  fprintf(1, 'G2 effect\n%1.8f\t%1.8f\n', sum(ch2(:,1)), sum(ch2(:,2)));
end