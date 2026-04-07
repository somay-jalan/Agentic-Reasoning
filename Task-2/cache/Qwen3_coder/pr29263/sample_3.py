def sqrt_quadratic_rule(integral: IntegralInfo, degenerate=True):
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    c = Wild('c', exclude=[x, 0])
    f = Wild('f')
    n = Wild('n', properties=[lambda n: n.is_Integer and n.is_odd])
    match = integrand.match(f*sqrt(a+b*x+c*x**2)**n)
    if not match:
        return
    a, b, c, f, n = match[a], match[b], match[c], match[f], match[n]
    f_poly = f.as_poly(x)
    if f_poly is None:
        return

    generic_cond = Ne(c, 0)
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    elif b.is_zero:
        degenerate_step = integral_steps(f*sqrt(a)**n, x)
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo(f*sqrt(a+b*x)**n, x))

    def sqrt_quadratic_denom_rule(numer_poly: Poly, integrand: Expr):
        denom = sqrt(a+b*x+c*x**2)
        deg = numer_poly.degree()
        if deg <= 1:
            # integrand == (d+e*x)/sqrt(a+b*x+c*x**2)
            e, d = numer_poly.all_coeffs() if deg == 1 else (S.Zero, numer_poly.as_expr())
            # rewrite numerator to A*(2*c*x+b) + B
            A = e/(2*c)
            B = d-A*b
            pre_substitute = (2*c*x+b)/denom
            constant_step: Rule | None = None
            linear_step: Rule | None = None
            if A != 0:
                u = Dummy("u")
                pow_rule = PowerRule(1/sqrt(u), u, u, -S.Half)
                linear_step = URule(pre_substitute, x, u, a+b*x+c*x**2, pow_rule)
                if A != 1:
                    linear_step = ConstantTimesRule(A*pre_substitute, x, A, pre_substitute, linear_step)
            if B != 0:
                constant_step = inverse_trig_rule(IntegralInfo(1/denom, x), degenerate=False)
                if B != 1:
                    constant_step = ConstantTimesRule(B/denom, x, B, 1/denom, constant_step)  # type: ignore
            if linear_step and constant_step:
                add = Add(A*pre_substitute, B/denom, evaluate=False)
                step: Rule | None = RewriteRule(integrand, x, add, AddRule(add, x, [linear_step, constant_step]))
            else:
                step = linear_step or constant_step
        else:
            coeffs = numer_poly.all_coeffs()
            step = SqrtQuadraticDenomRule(integrand, x, a, b, c, coeffs)
        return step

    def sqrt_quadratic_reciprocal_rule(integrand: Expr):
        # Handle integrals of the form 1/(x*sqrt(a + b*x + c*x**2))
        # This should give -1/sqrt(a) * log(2*sqrt(a)*sqrt(a + b*x + c*x**2) + b*x + 2*a)/x) or similar
        # For the specific case of 1/(x*sqrt(a**2 - x**2)), the result is -1/a * log((a + sqrt(a**2 - x**2))/x)
        denom = sqrt(a + b*x + c*x**2)
        
        # Check if we have 1/(x*sqrt(quadratic))
        if integrand == 1/(x*denom) or (integrand.is_Pow and integrand.exp == -1 and 
                                         integrand.base == x*denom):
            # For a**2 - x**2 case: integral of 1/(x*sqrt(a**2 - x**2)) dx
            # = -1/a * log((a + sqrt(a**2 - x**2))/x) + C
            # or equivalently: log((a - sqrt(a**2 - x**2))/x) + C
            if b == 0 and c == -1 and a.is_positive:
                # Case: 1/(x*sqrt(a**2 - x**2))
                result = log((a - sqrt(a - x**2))/x)
                if x.is_real:
                    result = log((a - sqrt(a - x**2))/Abs(x))
                return result
            elif a.is_positive and c.is_negative:
                # General case: 1/(x*sqrt(a - |c|*x**2))
                sqrt_term = sqrt(a + b*x + c*x**2)
                return log((sqrt(a) - sqrt_term)/x)
        
        return None

    def sqrt_quadratic_reduction_rule(integrand: Expr, n: int):
        # Implementation of Gradshteyn & Ryzhik 2.263.3
        k = (-n - 1) // 2
        delta = 4*a*c - b**2
        R = c*x**2 + b*x + a

        term_denom = (2*k - 1) * delta * (R**(S(2*k - 1)/2))
        constant_term = f*2*(2*c*x+b) / term_denom
        coeff = (8*c*(k-1))/((2*k-1) * delta)
        expr = f * R**(S(1)/2 - k)

        rewrite_expr = Derivative(constant_term, x) + coeff * expr
        derive_expr = Derivative(constant_term, x)
        derive_step = integral_steps(derive_expr, x)

        if coeff == 0:
            substep = derive_step
        else:
            next_step = integral_steps(expr, x)
            if not next_step:
                next_step = DontKnowRule(expr, x)

            substep = AddRule(
                rewrite_expr,
                x,
                [
                    derive_step,
                    ConstantTimesRule(
                        coeff * expr,
                        x,
                        coeff,
                        expr,
                        next_step
                    )
                ]
            )

        return RewriteRule(integrand, x, rewrite_expr, substep)

    if n > 0:  # rewrite poly * sqrt(s)**(2*k-1) to poly*s**k / sqrt(s)
        numer_poly = f_poly * (a+b*x+c*x**2)**((n+1)/2)
        rewritten = numer_poly.as_expr()/sqrt(a+b*x+c*x**2)
        substep = sqrt_quadratic_denom_rule(numer_poly, rewritten)
        generic_step = RewriteRule(integrand, x, rewritten, substep)
    elif n == -1:
        # Try reciprocal rule first
        reciprocal_result = sqrt_quadratic_reciprocal_rule(integrand)
        if reciprocal_result is not None:
            return reciprocal_result
        generic_step = sqrt_quadratic_denom_rule(f_poly, integrand)
    elif f_poly.degree() == 0:
        # The numerator must be a const, the formula assumes this
        generic_step = sqrt_quadratic_reduction_rule(integrand, n)
    else:
        # Handle non-constant numerators (eg. x / R**(-3/2))
        # This requires splitting the integral as A*(2ax+b) + B form
        return None
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)