for gr=1:2
  for i=0:10
    SGD(i / 10, gr, 'hid', '1m');
    SGD(i / 10, gr, 'ran', '1m');
    SGD(i / 10, gr, 'add', '1m');
  end
end