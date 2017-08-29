dataset = '1m';
privacy = 'udp';

for div=[1,2,4,8]
    preSGD([0, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, '1');
    preSGD([0, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, '2');
    preSGD([0, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    
    preSGD([0.1, 0], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([1, 0],   [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([10, 0],  [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');

    preSGD([0, 0.1], [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([0, 1],   [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
    preSGD([0, 10],  [privacy; privacy], [1/(1+div),div/(div+1)], dataset, 'all');
end