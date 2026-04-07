def sqrt_ratio_rule(integral):
    integrand, symbol = integral
    
    # Look for patterns like sqrt((a - x)/(a + x)) or similar ratios
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    
    # Pattern for sqrt((a - x)/(a + x))
    pattern1 = sqrt((a - symbol)/(a + symbol))
    
    match = integrand.match(pattern1 / symbol)
    if match and match[a].is_constant(symbol):
        a_val = match[a]
        # Rewrite sqrt((a-x)/(a+x))/x as (a-x)/(x*sqrt(a^2-x^2))
        rewritten = (a_val - symbol) / (symbol * sqrt(a_val**2 - symbol**2))
        if rewritten != integrand:
            substep = integral_steps(rewritten, symbol)
            if substep:
                return RewriteRule(integrand, symbol, rewritten, substep)
    
    # Also try the pattern sqrt((a + x)/(a - x))/x
    pattern2 = sqrt((a + symbol)/(a - symbol))
    match = integrand.match(pattern2 / symbol)
    if match and match[a].is_constant(symbol):
        a_val = match[a]
        # Rewrite sqrt((a+x)/(a-x))/x as (a+x)/(x*sqrt(a^2-x^2))
        rewritten = (a_val + symbol) / (symbol * sqrt(a_val**2 - symbol**2))
        if rewritten != integrand:
            substep = integral_steps(rewritten, symbol)
            if substep:
                return RewriteRule(integrand, symbol, rewritten, substep)

# Add sqrt_ratio_rule to the integrator
# We need to modify the integral_steps function to include this rule
# Specifically, we should add it to the do_one sequence for Mul expressions