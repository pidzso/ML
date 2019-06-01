shrink_user = in;
rng(shrink_user);

fprintf(1, 'Loading...\n');
load('data/netflix/netflix_all.mat');
  
% actual shrinking
x = full(train_vec(:, 1));
y = full(train_vec(:, 2));
z = full(train_vec(:, 3));
  
fprintf(1, 'Filtering users...\n');
% chose random users
rem_user = 1:max(x);
rem_user = rem_user(randperm(max(x)));
rem_user = rem_user(1:floor(max(x) * shrink_user / 100));
rem_x = ismember(x, rem_user) .* x;
  
% leftovers users
%left_y = not(ismember(x, rem_user)) .* x;
%train_vec_left = [left_x, y, z];
%train_vec_left_aux = train_vec_left;
%train_vec_left(any(train_vec_left_aux==0,2),:) = [];
%save('data/netflix/leftover.mat', '-v7.3', 'train_vec_left');
  
% shrink dataset
train_vec = [rem_x, y, z];
rem_train_vec = train_vec;
rem_train_vec(any(train_vec==0,2),:) = [];
train_vec = rem_train_vec;

clearvars -except train_vec shrink_user

fprintf(1, 'Recounting items...\n');
uniQ = sort(unique(train_vec(:,2)));
all = train_vec(:, 2);
for i = 1:size(uniQ)
    all( all == uniQ(i) ) = i;
end
train_vec(:, 2) = all;

fprintf(1, 'Recounting users...\n');
uniQ = sort(unique(train_vec(:,1)));
all = train_vec(:, 1);
for i = 1:size(uniQ)
    all( all == uniQ(i) ) = i;
end
train_vec(:, 1) = all;
  
n_item   = max(train_vec(:, 1));
n_user   = max(train_vec(:, 2));
n_rating = size(train_vec, 1);

fprintf(1, 'Preprocessing...\n');
[pr_d, pr_i, pr_u, pr_r, avg_u, avg_i] = preproc(train_vec, 10);
pr_d(:, 3) = bounding(pr_d(:, 3), 2);
save(strcat('data/netflix/proc_netflix', int2str(shrink_user), '.mat'), 'pr_d', 'pr_i', 'pr_u', 'pr_r', 'avg_u', 'avg_i');