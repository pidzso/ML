div = [5/10, 5/10];
preSGD([0, 0], ['pgd'; 'pgd'], div, '1m', '1');
preSGD([0, 0], ['pgd'; 'pgd'], div, '1m', '2');
preSGD([0, 0], ['pgd'; 'pgd'], div, '1m', 'all');
preSGD([log(10), 0], ['pgd'; 'pgd'], div, '1m', 'all');
preSGD([0, log(10)], ['pgd'; 'pgd'], div, '1m', 'all');
preSGD([log(10), log(10)], ['pgd'; 'pgd'], div, '1m', 'all');