function [lambda, max_iter, n_features, n_group, epsilon, ...
          bound_f, bound_avg, bound_rat, ...
          s_type, toleration, buffer, stopped] = param(dat)
  
  lambda     = 0.01;     % regularization
  max_iter   = 20;       % iteration
  n_features = 4;        % number of features
  bound_f    = 0.5;      % feature bound
  bound_avg  = 2;        % user average bound
  bound_rat  = 2;        % rating bound
  n_group    = 2;        % number of groups
  epsilon    = 7.5/1000; % learning rate
  rmse       = zeros(max_iter, n_group, n_group); % group error
  
  % stopping parameters
  s_type     = ['siz'; 'siz'; 'siz'; 'dir'; 'dir'; 'dir'];
  toleration = [0, 0, 2, 5];
  buffer     = [2, 5, 2, 5];
  stopped    = [0, 0, 0, 0; 0, 0, 0, 0];
end  