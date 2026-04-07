M = M.applyfunc(
    lambda x: nsimplify(x, rational=True) if x.has(Float) else x)