function ind = stop(type, toleration, iter, buffer, rmse, g)
  init  = 10;  % skipping the first x iteration
  ind   = 0;
  other = 3 - g;
  if iter > buffer + init;
    buff = iter-buffer + 1;
    ch   = squeeze(rmse(:, g, other));
  
    switch type
      case 'dir'
        aux  = sign(ch(buff:iter));
        if abs(sum(aux(aux < 0))) >= toleration
          ind = 1;
        end
        clear aux;
      
      case 'siz'
        if sum(ch(buff:iter)) <= toleration
          ind = 1;
        end
    end    
  end