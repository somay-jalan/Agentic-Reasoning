if A and B:
    M = Matrix([[A, B], [C, D]])
else:
    if A or B:
        raise ValueError("must give A and B")
    # no constraints given
    M = Matrix([[C, D]])