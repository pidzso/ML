load('data/10m/10m.mat')

x = full(train_vec(:, 1));
y = full(train_vec(:, 2));
z = full(train_vec(:, 3));

[a, aa] = hist(x, unique(x));
[b, bb] = hist(y, unique(y));

% remove outliers
ind = 0;
while ind == 1
  aux = size(x);
  for i=1:size(a, 2)
    if or(a(i)<50, a(i)>500)
      x(x == aa(i)) = 0;
    end
  end
  for i=1:size(x)
    if x(i) == 0
      y(i) = 0;
      z(i) = 0;
    end
  end
  for i=1:size(b, 2)
    if or(b(i)<100, b(i)>1000)
      y(y == bb(i)) = 0;
    end
  end
  for i=1:size(y)
    if y(i) == 0
      x(i) = 0;
      z(i) = 0;
    end
  end
  x(x == 0) = [];
  y(y == 0) = [];
  z(z == 0) = [];
  if aux == size(x)
    ind = 0;
  end
end
%train_vec = [x, y, z];

% chose random users
shrink_user = 0.33;
rem_user = 1:max(x);
rem_user = rem_user(randperm(max(x)));
rem_user = rem_user(1:floor(max(x) * shrink_user));
rem_x = ismember(x, rem_user) .* x;

% chose random items
shrink_item = 0.75;
rem_item = 1:max(y);
rem_item = rem_item(randperm(max(y)));
rem_item = rem_item(1:floor(max(y) * shrink_item));
rem_y = ismember(y, rem_item) .* y;

% shrink dataset
train_vec = [rem_x, rem_y, z];
rem_train_vec = train_vec;
rem_train_vec(any(train_vec==0,2),:) = [];
train_vec = rem_train_vec;

uniQ = sort(unique(train_vec(:,2)));
all = train_vec(:, 2);
for i = 1:size(uniQ)
    all( all == uniQ(i) ) = i;
end
train_vec(:, 2) = all;

uniQ = sort(unique(train_vec(:,1)));
all = train_vec(:, 1);
for i = 1:size(uniQ)
    all( all == uniQ(i) ) = i;
end
train_vec(:, 1) = all;

n_user   = max(train_vec(:, 1));
n_item   = max(train_vec(:, 2));
n_rating = size(train_vec, 1);