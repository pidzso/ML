function new = bounding(vec, bound)
  new = vec;
  new(new > bound) = bound;
  new(new < -bound) = -bound;
end