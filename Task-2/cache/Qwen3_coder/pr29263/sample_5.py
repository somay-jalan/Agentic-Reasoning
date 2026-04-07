def sqrt_fraction_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol])
    d = Wild('d', exclude=[symbol])
    
    # Pattern: sqrt((a + b*x)/(c + d*x))/x
    pattern = sqrt((a + b*symbol)/(c + d*symbol))/symbol
    
    match = integrand.match(pattern)
    if match:
        a_val, b_val, c_val, d_val = match[a], match[b], match[c], match[d]
        
        # Check if this is the specific form sqrt((a - x)/(a + x))/x
        # which means b = -1, d = 1, a = c (so a - x and a + x)
        if (b_val == -1 and d_val == 1 and a_val == c_val and a_val != 0):
            # For sqrt((a - x)/(a + x))/x, use the known result
            # = ln((a - sqrt(a^2 - x^2))/|x|) - arcsin(x/a) + C
            result = log((a_val - sqrt(a_val**2 - symbol**2))/Abs(symbol)) - asin(symbol/a_val)
            return result
    
    # More general case: sqrt((a + b*x)/(c + d*x))/x
    # This is more complex and may need case analysis
    return None