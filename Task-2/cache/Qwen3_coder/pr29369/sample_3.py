else:
    aux = -A.cols  # set so -aux will give all cols below

o, p, d = _simplex(A, b, C)
return o, p[:-aux]  # don't include aux values