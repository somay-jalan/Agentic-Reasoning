def bode_magnitude_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', **kwargs):
    """
    Returns the numerical data of the Bode magnitude plot of the system.
    It is internally used by ``bode_magnitude_plot`` to get the data
    for plotting Bode magnitude plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode magnitude plot.
        y = y-axis values of the Bode magnitude plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_magnitude_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_magnitude_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.5148378120533502e-05,..., 68437.36188804005, 100000.0],
    [-6.020599914256786, -6.0205999155219505,..., -193.4117304087953, -200.00000000260573])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    from sympy.functions import re, im, sqrt
    
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    # Use re() and im() instead of Abs() to avoid nan values
    mag_re = re(w_expr)
    mag_im = im(w_expr)
    mag = sqrt(mag_re**2 + mag_im**2)
    mag = 20*log(mag, 10)

    x, y = LineOver1DRangeSeries(mag,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    return x, y