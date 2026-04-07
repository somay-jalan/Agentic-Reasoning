def _simplex(A, B, C, D=None, dual=False):
    """Return ``(o, x, y)`` obtained from the two-phase simplex method
    using Bland's rule: ``o`` is the minimum value of primal,
    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and
    the maximum of the dual, ``y^{T}B - D``, under constraints
    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of
    the system, pass `dual=True` and ``(o, y, x)`` will be returned.

    Note: the nonnegative constraints for ``x`` and ``y`` supercede
    any values of ``A`` and ``B`` that are inconsistent with that
    assumption, so if a constraint of ``x >= -1`` is represented
    in ``A`` and ``B``, no value will be obtained that is negative; if
    a constraint of ``x <= -1`` is represented, an error will be
    raised since no solution is possible.

    This routine relies on the ability of determining whether an
    expression is 0 or not. This is guaranteed if the input contains
    only Float or Rational entries. It will raise a TypeError if
    a relationship does not evaluate to True or False.

    Examples
    ========

    >>> from sympy.solvers.simplex import _simplex
    >>> from sympy import Matrix

    Consider the simple minimization of ``f = x + y + 1`` under the
    constraint that ``y + 2*x >= 4``. This is the "standard form" of
    a minimization.

    In the nonnegative quadrant, this inequality describes a area above
    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum
    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard
    minimization:

    >>> A = Matrix([[2, 1]])
    >>> B = Matrix([4])
    >>> C = Matrix([[1, 1]])
    >>> D = Matrix([-1])

    Confirm that this is the system of interest:

    >>> from sympy.abc import x, y
    >>> X = Matrix([x, y])
    >>> (C*X - D)[0]
    x + y + 1
    >>> [i >= j for i, j in zip(A*X, B)]
    [2*x + y >= 4]

    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:

    >>> _simplex(-A, -B, C, D)
    (3, [2, 0], [1/2])

    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for
    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the
    transpose of the matrix representation of the standard minimization:

    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]
    >>> a, b, c, d = tr(A, B, C, D)

    This time ``a*x <= b`` is the expected inequality for the `_simplex`
    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be
    changed (so that minimizing the negative will give the negative of
    the maximum of ``F``):

    >>> _simplex(a, b, -c, -d)
    (-3, [1/2], [2, 0])

    The negative of ``F`` and the min of ``f`` are the same. The dual
    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``
    under constraints a*x <= b``:

    >>> y = Matrix(['y'])
    >>> (c*y - d)[0]
    4*y + 1
    >>> [i <= j for i, j in zip(a*y,b)]
    [2*y <= 1, y <= 1]

    In this 1-dimensional dual system, the more restrictive constraint is
    the first which limits ``y`` between 0 and 1/2 and the maximum of
    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.

    In this case the values for ``x`` and ``y`` were the same when the
    dual representation was solved. This is not always the case (though
    the value of the function will be the same).

    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]
    >>> A, B, C, D = [Matrix(i) for i in l]
    >>> _simplex(A, B, -C, -D)
    (-6, [3, 2], [1, 0, 0, 0])
    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]
    (-6, [1, 0, 0, 0], [5, 0])

    In both cases the function has the same value:

    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])
    True

    See Also
    ========
    _lp - poses min/max problem in form compatible with _simplex
    lpmin - minimization which calls _lp
    lpmax - maximimzation which calls _lp

    References
    ==========

    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction
           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf

    """
    A, B, C, D = [Matrix(i) for i in (A, B, C, D or [0])]
    if dual:
        _o, d, p = _simplex(-A.T, C.T, B.T, -D)
        return -_o, d, p

    if A and B:
        M = Matrix([[A, B], [C, D]])
    else:
        if A or B:
            raise ValueError("must give A and B")
        # no constraints given
        M = Matrix([[C, D]])
    n = M.cols - 1
    m = M.rows - 1

    if not all(i.is_Float or i.is_Rational for i in M):
        # with literal Float and Rational we are guaranteed the
        # ability of determining whether an expression is 0 or not
        raise TypeError(filldedent("""
            Only rationals and floats are allowed.
            """
            )
        )

    # x variables have priority over y variables during Bland's rule
    # since False < True
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]

    # Phase 1: find a feasible solution or determine none exist

    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        if all(B[i] >= 0 for i in range(B.rows)):
            # We have found a feasible solution
            break

        # Find k: first row with a negative rightmost entry
        for k in range(B.rows):
            if B[k] < 0:
                break  # use current value of k below
        else:
            pass  # error will raise below

        # Choose pivot column, c
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            raise InfeasibleLPError(filldedent("""
                The constraint set is empty!"""))
        _, c = min((X[i], i) for i in piv_cols) # Bland's rule

        # Choose pivot row, r
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] >= 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    # Phase 2: from a feasible solution, pivot to optimal
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]

        # Choose a pivot column, c
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        _, c = min((X[i], i) for i in piv_cols)  # Bland's rule

        # Choose a pivot row, r
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent("""
                Objective function can assume
                arbitrarily large values!"""))
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    argmax = [None] * n
    argmin_dual = [None] * m

    for i, (v, n) in enumerate(X):
        if v == False:
            argmax[n] = S.Zero
        else:
            argmin_dual[n] = M[-1, i]

    for i, (v, n) in enumerate(Y):
        if v == True:
            argmin_dual[n] = S.Zero
        else:
            argmax[n] = M[i, -1]

    return -M[-1, -1], argmax, argmin_dual