function [gr_u, gr_size, gr_t_size, gr_v_size, group] = groupping(n_user, n_group, gr_div, train_vec)
  % variables
  n_gr_u    = [floor(n_user / 2), n_user - floor(n_user / 2)]; % user population within groups
  group     = cell(n_group, 2);    % groupped ratings
  gr_size   = zeros(n_group, 1);   % group sizes
  gr_t_size = zeros(n_group, 1);   % group training size
  gr_v_size = zeros(n_group, 1);   % group verification size
  
  gr_u   = [zeros(n_user, 1), zeros(n_user, 1)];
  r_perm = randperm(n_user);
  %r_perm = 1:n_user;
  
  gr_u(1:n_gr_u(1), 1) = transpose(r_perm(1:n_gr_u(1)));
  gr_u(1:n_gr_u(2), 2) = transpose(r_perm((n_gr_u(1)+1):n_user));

  for g = 1:n_group
    aux = zeros(size(train_vec, 1), 3);
  
    % creating a group
    buff = 0;
    for j = 1:n_gr_u(g)
      aux_loc = train_vec(train_vec(:,1) == r_perm((g-1) * n_gr_u(1) + j),:);
      for i=1:size(aux_loc, 1)
        aux(buff + i, :) = aux_loc(i, :);
      end
      buff = buff + size(aux_loc, 1);
    end
    aux( ~any(aux, 2), : ) = [];

    % local variables
    aux_perm   = randperm(size(aux, 1));
    aux        = aux(aux_perm,:);
    gr_size(g) = size(aux, 1);
  
    % training - verification set
    gr_t_size(g) = floor(gr_size(g) * 0.8);
    gr_v_size(g) = gr_size(g) - gr_t_size(g);
    aux_tr       = aux(1:gr_t_size(g), :);
    aux_v        = aux(gr_t_size(g) + 1:end, :);
    group(g, 1)  = mat2cell(aux_tr, gr_t_size(g), 3);
    group(g, 2)  = mat2cell(aux_v, gr_v_size(g), 3);
  end
end