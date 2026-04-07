def _eigenvals_dict(
        M: MatrixBase,
        error_when_incomplete: bool = True,
        simplify: Callable[[Expr], Expr] | bool = False,
        rational: bool = False,
        **flags: Any,
    ) -> dict[Expr, int]:

    iblocks = M.strongly_connected_components()
    all_eigs: dict[Expr, int] = {}
    expanded_eigs: dict[Expr, Expr] = {}

    # XXX: Only RepMatrix has _rep ...
    is_dom = M._rep.domain in (ZZ, QQ) # type: ignore
    for b in iblocks:

        # Fast path for a 1x1 block:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs[val] = all_eigs.get(val, 0) + 1
            continue

        block = M[b, b]

        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()

        factors = charpoly.factor_list()[1]

        for factor, multiplicity in factors:
            eigs = roots(factor, multiple=False, **flags)

            degree = int(factor.degree())
            if sum(eigs.values()) != degree:
                try:
                    eigs = dict(factor.all_roots(multiple=False))
                except NotImplementedError:
                    if error_when_incomplete:
                        raise MatrixError(eigenvals_error_message)
                    else:
                        eigs = {}

            for k, v in eigs.items():
                # Try a bit to canonicalize the eigenvalue expressions to get
                # the multiplicity correct. This is not robust enough in general
                # if different subroutines in roots can return different forms
                # for the same root.
                k_expanded = k.expand()
                if k_expanded in expanded_eigs:
                    k = expanded_eigs[k_expanded]
                else:
                    expanded_eigs[k_expanded] = k

                v_total = v * multiplicity
                if k in all_eigs:
                    all_eigs[k] += v_total
                else:
                    all_eigs[k] = v_total

    if not rational:
        if not simplify:
            return all_eigs
        if not isinstance(simplify, FunctionType):
            simplify = _simplify
        return {simplify(key): value for key, value in all_eigs.items()}
    
    # When rational=True, convert eigenvalues to rational form
    from sympy.simplify import nsimplify
    rational_eigs = {}
    for key, value in all_eigs.items():
        # Convert the eigenvalue to rational if it's a Float
        if isinstance(key, Float):
            rational_val = nsimplify(key, rational=True)
        else:
            rational_val = key
        # Apply simplification if requested
        if simplify:
            if not isinstance(simplify, FunctionType):
                simplify_func = _simplify
            else:
                simplify_func = simplify
            rational_val = simplify_func(rational_val)
        rational_eigs[rational_val] = value
    
    return rational_eigs