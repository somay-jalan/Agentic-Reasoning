if not A:
    if b:
        raise ValueError("A and b must both be given")
    # the governing equations will be simple constraints
    # on variables
    A, b = zeros(0, C.cols), zeros(C.cols, 1)