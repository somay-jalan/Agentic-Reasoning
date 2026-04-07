def _eval_derivative_n_times(self, x, n):
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy import sin, cos
    n = sympify(n)
    if n.is_integer and n.is_nonnegative:
        arg = self.args[0]
        # d^n/dx^n sin(x) = sin(x + n*pi/2)
        return sin(arg + n*pi/2)
    return None

# Add this method to sin class
sin._eval_derivative_n_times = _eval_derivative_n_times


def _eval_derivative_n_times(self, x, n):
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy import sin, cos
    n = sympify(n)
    if n.is_integer and n.is_nonnegative:
        arg = self.args[0]
        # d^n/dx^n cos(x) = cos(x + n*pi/2)
        return cos(arg + n*pi/2)
    return None

# Add this method to cos class
cos._eval_derivative_n_times = _eval_derivative_n_times