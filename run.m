rng(in); 
format long;

p    = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
e    = 1./p-1;
e(1) = 0;

baseline_init      = [0, 0];
resSup_init        = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
resDP_init         = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
resSup_init(:,:,2) = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
resDP_init(:,:,2)  = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];

fprintf(1, 'Real Groupping...\n');
load(strcat('data/1m/proc_1m', '.mat'));
[gr_u, gr_size, gr_t_size, gr_v_size, group] = groupping(proc_u, 2, [0.5,0.5], proc_d);
save(strcat('data/1m/', int2str(in), 'group0.mat'), 'gr_u', 'gr_size', 'gr_t_size', 'gr_v_size', 'group');

%load(strcat('data/1m/', int2str(in), 'group0.mat'));

fprintf(1, 'Real Baseline...\n');
b = 3;
for a=1:b
  [x, y, act, init] = preSGD([0,0], ['sup'; 'sup'], [0.5, 0.5], in*10, '1');
  baseline_init(1) = baseline_init(1) + (init(1) - act(1)) / b;
  [x, y, act, init] = preSGD([0,0], ['sup'; 'sup'], [0.5, 0.5], in*10, '2');
  baseline_init(2) = baseline_init(2) + (init(2) - act(2)) / b;
end

fprintf(1, 'Real bDP...\n');
met = 'bdp';
b = 3;
for a=1:b
  for i=e
    for j=e
      [x, y, act, init] = preSGD([i, j], [met; met], [0.5, 0.5], in*10, 'all');
      resDP_init(find(e==i), find(e==j), 1) = resDP_init(find(e==i), find(e==j), 1) + (init(1) - act(1)) / b;
      resDP_init(find(e==j), find(e==i), 2) = resDP_init(find(e==j), find(e==i), 2) + (init(2) - act(2)) / b;
    end
  end
end

fprintf(1, 'Real Sup...\n');
met = 'sup';
b = 3;
for a=1:b
  for i=p
    for j=p
      [x, y, act, init] = preSGD([i, j], [met; met], [0.5, 0.5], in*10, 'all');
      resSup_init(find(p==i), find(p==j), 1) = resSup_init(find(p==i), find(p==j), 1) + (init(1) - act(1)) / b;
      resSup_init(find(p==j), find(p==i), 2) = resSup_init(find(p==j), find(p==i), 2) + (init(2) - act(2)) / b;
    end
  end
end

resSup_init(:,:,1) = (resSup_init(:,:,1) - baseline_init(1)) / baseline_init(1);
resSup_init(:,:,2) = (resSup_init(:,:,2) - baseline_init(2)) / baseline_init(2);
resDP_init(:,:,1)  = (resDP_init(:,:,1) - baseline_init(1)) / baseline_init(1);
resDP_init(:,:,2)  = (resDP_init(:,:,2) - baseline_init(2)) / baseline_init(2);

fprintf(1, 'Init Sup1/DP1/Sup2/DP2...\n');
disp(resSup_init(:,:,1));
disp(resDP_init(:,:,1));
disp(resSup_init(:,:,2));
disp(resDP_init(:,:,2));

finSup1_init        = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finDP1_init         = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finSup1_init(:,:,2) = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finDP1_init(:,:,2)  = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];

finSup2_init        = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finDP2_init         = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finSup2_init(:,:,2) = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
finDP2_init(:,:,2)  = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];

b = 3;
for a=1:b
  fprintf(1, 'Round\t%i\n', a);
  for subgroup=1:2

    baseline_init      = [0, 0];
    resSup_init        = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
    resDP_init         = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
    resSup_init(:,:,2) = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];
    resDP_init(:,:,2)  = [[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0];[0,0,0,0,0,0]];

    fprintf(1, 'Fake Groupping...\n');
    shrink(in, subgroup, a);

    fprintf(1, 'Fake Baseline...\n');
    bb = 3;
    for aa=1:bb
      [x, y, act, init] = preSGD([0,0], ['sup'; 'sup'], [0.5, 0.5], in*10+subgroup, '1');
      baseline_init(1)  = baseline_init(1) + (init(1) - act(1)) / bb;
      [x, y, act, init] = preSGD([0,0], ['sup'; 'sup'], [0.5, 0.5], in*10+subgroup, '2');
      baseline_init(2)  = baseline_init(2) + (init(2) - act(2)) / bb;
    end

    fprintf(1, 'Fake bDP...\n');
    met = 'bdp';
    bb = 3;
    for aa=1:bb
      for i=e
        for j=e
          [x, y, act, init]                     = preSGD([i, j], [met; met], [0.5, 0.5], in*10+subgroup, 'all');
          resDP_init(find(e==i), find(e==j), 1) = resDP_init(find(e==i), find(e==j), 1) + (init(1) - act(1)) / bb;
          resDP_init(find(e==j), find(e==i), 2) = resDP_init(find(e==j), find(e==i), 2) + (init(2) - act(2)) / bb;
        end
      end
    end

    fprintf(1, 'Fake Sup...\n');
    met = 'sup';
    bb = 3;
    for aa=1:bb
      for i=p
        for j=p
          [x, y, act, init]                      = preSGD([i, j], [met; met], [0.5, 0.5], in*10+subgroup, 'all');
          resSup_init(find(p==i), find(p==j), 1) = resSup_init(find(p==i), find(p==j), 1) + (init(1) - act(1)) / bb;
          resSup_init(find(p==j), find(p==i), 2) = resSup_init(find(p==j), find(p==i), 2) + (init(2) - act(2)) / bb;
        end
      end
    end

    resSup_init(:,:,1) = (resSup_init(:,:,1) - baseline_init(1)) / baseline_init(1);
    resSup_init(:,:,2) = (resSup_init(:,:,2) - baseline_init(2)) / baseline_init(2);
    resDP_init(:,:,1)  = (resDP_init(:,:,1) - baseline_init(1)) / baseline_init(1);
    resDP_init(:,:,2)  = (resDP_init(:,:,2) - baseline_init(2)) / baseline_init(2);

    if subgroup == 1
      finSup1_init(:,:,1) = finSup1_init(:,:,1) + resSup_init(:,:,1) / b;
      finSup1_init(:,:,2) = finSup1_init(:,:,2) + resSup_init(:,:,2) / b;
      finDP1_init(:,:,1)  = finDP1_init(:,:,1) + resDP_init(:,:,1) / b;
      finDP1_init(:,:,2)  = finDP1_init(:,:,2) + resDP_init(:,:,2) / b;
    else
      finSup2_init(:,:,1) = finSup2_init(:,:,1) + resSup_init(:,:,1) / b;
      finSup2_init(:,:,2) = finSup2_init(:,:,2) + resSup_init(:,:,2) / b;
      finDP2_init(:,:,1)  = finDP2_init(:,:,1) + resDP_init(:,:,1) / b;
      finDP2_init(:,:,2)  = finDP2_init(:,:,2) + resDP_init(:,:,2) / b;
    end
  end
end

fSu1_init = (finSup1_init(:,:,1) + finSup1_init(:,:,2)) / 2;
fDP1_init = (finDP1_init(:,:,1) + finDP1_init(:,:,2)) / 2;
fSu2_init = (finSup2_init(:,:,1) + finSup2_init(:,:,2)) / 2;
fDP2_init = (finDP2_init(:,:,1) + finDP2_init(:,:,2)) / 2;

disp(fSu1_init);
disp(fDP1_init);
disp(fSu2_init);
disp(fDP2_init);

clear;