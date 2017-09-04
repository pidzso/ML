dataset = 'netflix';
privacy = 'udp';

for div=[2,8]
    preSGD([1/0.1-1, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([1/0.5-1, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([1/0.9-1, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');

    preSGD([0, 1/0.1-1], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([0, 1/0.5-1], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([0, 1/0.9-1], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
end