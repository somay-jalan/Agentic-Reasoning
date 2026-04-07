def _eigenvals(M,
        error_when_incomplete: bool = True,
        *,
        simplify: Callable[[Expr], Expr] | bool = False,
        multiple: bool = False,
        rational: bool = False,
        **flags,
    ) -> dict[Expr, int] | list[Expr]:
    r"""Compute eigenvalues of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        If it is set to ``True``, it will raise an error if not all
        eigenvalues are computed. This is caused by ``roots`` not returning
        a full list of eigenvalues.

    simplify : bool or function, optional
        If it is set to ``True``, it attempts to return the most
        simplified form of expressions returned by applying default
        simplification method in every routine.

        If it is set to ``False``, it will skip simplification in this
        particular routine to save computation resources.

        If a function is passed to, it will attempt to apply
        the particular function as simplification method.

    rational : bool, optional
        If it is set to ``True``, every floating point numbers would be
        replaced with rationals before computation. It can solve some
        issues of ``roots`` routine not working well with floats.

    multiple : bool, optional
        If it is set to ``True``, the result will be in the form of a
        list.

        If it is set to ``False``, the result will be in the form of a
        dictionary.

    Returns
    =======

    eigs : list or dict
        Eigenvalues of a matrix. The return format would be specified by
        the key ``multiple``.

    Raises
    ======

    MatrixError
        If not enough roots got computed.

    NonSquareMatrixError
        If attempted to compute eigenvalues from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1]
    >>> M.eigenvals()
    {-1: 1, 0: 1, 2: 1}

    See Also
    ========

    MatrixBase.charpoly
    eigenvects

    Notes
    =====

    Eigenvalues of a matrix $A$ can be computed by solving a matrix
    equation $\det(A - \lambda I) = 0$
    """
    if not M:
        if multiple:
            return []
        return {}

    if not M.is_square:
        raise NonSquareMatrixError("{} must be a square matrix.".format(M))

    if rational:
        from sympy.simplify import nsimplify
        M = M.applyfunc(
            lambda x: nsimplify(x, rational=True) if x.has(Float) else x)

    # ... rest of the function remains unchanged until the return statement

    if multiple:
        return _eigenvals_list(
            M, error_when_incomplete=error_when_incomplete, simplify=simplify,
            **flags)
    return _eigenvals_dict(
        M, error_when_incomplete=error_when_incomplete, simplify=simplify,
        **flags)