dataset = '75';

%e = [1/0.2-1,1/0.4-1,1/0.6-1,1/0.8-1];
%p = [  0.2,    0.4,    0.6,    0.8];

div = 0;
met = 'hid';

preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '1');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '2');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0.25, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

met='bdp';

preSGD([1/0.25-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

dataset = '5';

met = 'hid';

preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '1');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '2');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0.25, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

met='bdp';

preSGD([1/0.25-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

dataset = '25';

met = 'hid';

preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '1');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, '2');
preSGD([0, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0.25, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.25], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.5], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.75], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.25, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.5, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.75, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([0.99, 0.99], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

met='bdp';

preSGD([1/0.25-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 0], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.25-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.5-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.75-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');

preSGD([0, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.25-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.5-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.75-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
preSGD([1/0.99-1, 1/0.99-1], [met; met], [1/(1+div),div/(div+1)], dataset, 'all');
