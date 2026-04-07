def sqrt_rational_substitution_rule(integral):
    integrand, symbol = integral
    x = symbol
    
    # Handle sqrt((a-x)/(a+x))/x form
    a = Wild('a', exclude=[x])
    pattern = sqrt((a - x)/(a + x))/x
    
    match = integrand.match(pattern)
    if match and match[a]:
        a_val = match[a]
        # Rewrite as (a-x)/(x*sqrt(a^2-x^2))
        rewritten = (a_val - x)/(x*sqrt(a_val**2 - x**2))
        # Split into two terms: a/(x*sqrt(a^2-x^2)) - 1/sqrt(a^2-x^2)
        term1 = a_val/(x*sqrt(a_val**2 - x**2))
        term2 = 1/sqrt(a_val**2 - x**2)
        
        step1 = integral_steps(term1, x)
        step2 = integral_steps(term2, x)
        
        if step1 and step2:
            substep = AddRule(rewritten, x, [step1, ConstantTimesRule(-term2, x, -1, term2, step2)])
            return RewriteRule(integrand, x, rewritten, substep)
    
    # Also handle the form sqrt((a-x)/(a+x))*f(x) where f(x) is 1/x
    pattern2 = sqrt((a - x)/(a + x))
    match2 = integrand.match(pattern2/x)
    if match2 and match2[a]:
        a_val = match2[a]
        rewritten = (a_val - x)/(x*sqrt(a_val**2 - x**2))
        term1 = a_val/(x*sqrt(a_val**2 - x**2))
        term2 = 1/sqrt(a_val**2 - x**2)
        
        step1 = integral_steps(term1, x)
        step2 = integral_steps(term2, x)
        
        if step1 and step2:
            substep = AddRule(rewritten, x, [step1, ConstantTimesRule(-term2, x, -1, term2, step2)])
            return RewriteRule(integrand, x, rewritten, substep)

# Add this rule to the integration process