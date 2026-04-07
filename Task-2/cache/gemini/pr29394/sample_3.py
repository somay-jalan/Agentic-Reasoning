from __future__ import annotations
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
    cos_table, ipartfrac, fermat_coords)
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.core.expr import Expr


###############################################################################
########################## UTILITIES ##########################################
###############################################################################


def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """
    if isinstance(arg, Float):
        return None
    else:
        return arg.as_coefficient(S.ImaginaryUnit)

###############################################################################
########################## TRIGONOMETRIC FUNCTIONS ############################
###############################################################################


class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational and fuzzy_not(s.args[0].is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
            pi_coeff = _pi_coeff(self.args[0])
            if pi_coeff is not None and pi_coeff.is_rational:
                return True
        else:
            return s.is_algebraic

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit

    def _as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.args[0].expand(deep, **hints), S.Zero)
            else:
                return (self.args[0], S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (re, im)

    def _period(self, general_period, symbol=None):
        f = expand_mul(self.args[0])
        if symbol is None:
            symbol = tuple(f.free_symbols)[0]

        if not f.has(symbol):
            return S.Zero

        if f == symbol:
            return general_period

        if symbol in f.free_symbols:
            if f.is_Mul:
                g, h = f.as_independent(symbol)
                if h == symbol:
                    return general_period/abs(g)

            if f.is_Add:
                a, h = f.as_independent(symbol)
                g, h = h.as_independent(symbol, as_Add=False)
                if h == symbol:
                    return general_period/abs(g)

        raise NotImplementedError("Use the periodicity function instead.")


@cacheit
def _table2():
    # If nested sqrt's are worse than un-evaluation
    # you can require q to be in (1, 2, 3, 4, 6, 12)
    # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
    # expressions with 2 or fewer sqrt nestings.
    return {
        12: (3, 4),
        20: (4, 5),
        30: (5, 6),
        15: (6, 10),
        24: (6, 8),
        40: (8, 10),
        60: (20, 30),
        120: (40, 60)
    }


def _peeloff_pi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $\pi$.
    This assumes ARG to be an Add.
    The multiple of $\pi$ returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    """
    pi_coeff = S.Zero
    rest_terms = []
    for a in Add.make_args(arg):
        K = a.coeff(pi)
        if K and K.is_rational:
            pi_coeff += K
        else:
            rest_terms.append(a)

    if pi_coeff is S.Zero:
        return arg, S.Zero

    m1 = (pi_coeff % S.Half)
    m2 = pi_coeff - m1
    if m2.is_integer or ((2*m2).is_integer and m2.is_even is False):
        return Add(*(rest_terms + [m1*pi])), m2
    return arg, S.Zero


def _pi_coeff(arg: Expr, cycles: int = 1) -> Expr | None:
    r"""
    When arg is a Number times $\pi$ (e.g. $3\pi/2$) then return the Number
    normalized to be in the range $[0, 2]$, else `None`.

    When an even multiple of $\pi$ is encountered, if it is multiplying
    something with known parity then the multiple is returned as 0 otherwise
    as 2.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _pi_coeff
    >>> from sympy import pi, Dummy
    >>> from sympy.abc import x
    >>> _pi_coeff(3*x*pi)
    3*x
    >>> _pi_coeff(11*pi/7)
    11/7
    >>> _pi_coeff(-11*pi/7)
    3/7
    >>> _pi_coeff(4*pi)
    0
    >>> _pi_coeff(5*pi)
    1
    >>> _pi_coeff(5.0*pi)
    1
    >>> _pi_coeff(5.5*pi)
    3/2
    >>> _pi_coeff(2 + pi)

    >>> _pi_coeff(2*Dummy(integer=True)*pi)
    2
    >>> _pi_coeff(2*Dummy(even=True)*pi)
    0

    """
    if arg is pi:
        return S.One
    elif not arg:
        return S.Zero
    elif arg.is_Mul:
        cx = arg.coeff(pi)
        if cx:
            c, x = cx.as_coeff_Mul()  # pi is not included as coeff
            if c.is_Float:
                # recast exact binary fractions to Rationals
                f = abs(c) % 1
                if f != 0:
                    p = -int(round(log(f, 2).evalf()))
                    m = 2**p
                    cm = c*m
                    i = int(cm)
                    if equal_valued(i, cm):
                        c = Rational(i, m)
                        cx = c*x
                else:
                    c = Rational(int(c))
                    cx = c*x
            if x.is_integer:
                c2 = c % 2
                if c2 == 1:
                    return x
                elif not c2:
                    if x.is_even is not None:  # known parity
                        return S.Zero
                    return Integer(2)
                else:
                    return c2*x
            return cx
    elif arg.is_zero:
        return S.Zero
    return None


class sin(TrigonometricFunction):
    r"""
    The sine function.

    Returns the sine of x (measured in radians).

    Explanation
    ===========

    This function will evaluate automatically in the
    case $x/\pi$ is some rational number [4]_.  For example,
    if $x$ is a multiple of $\pi$, $\pi/2$, $\pi/3$, $\pi/4$, and $\pi/6$.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    sympy.functions.elementary.trigonometric.csc
    sympy.functions.elementary.trigonometric.cos
    sympy.functions.elementary.trigonometric.sec
    sympy.functions.elementary.trigonometric.tan
    sympy.functions.elementary.trigonometric.cot
    sympy.functions.elementary.trigonometric.asin
    sympy.functions.elementary.trigonometric.acsc
    sympy.functions.elementary.trigonometric.acos
    sympy.functions.elementary.trigonometric.asec
    sympy.functions.elementary.trigonometric.atan
    sympy.functions.elementary.trigonometric.acot
    sympy.functions.elementary.trigonometric.atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] https://mathworld.wolfram.com/TrigonometryAngles.html

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_derivative_n_times(self, x, n):
        return sin(self.args[0] + n*S.Pi/2) * self.args[0].diff(x)**n

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            from sympy.sets.sets import FiniteSet
            min, max = arg.min, arg.max
            d = floor(min/(2*pi))
            if min is not S.NegativeInfinity:
                min = min - d*2*pi
            if max is not S.Infinity:
                max = max - d*2*pi
            if AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet and \
                    AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2),
                        pi*Rational(7, 2))) is not S.EmptySet:
                return AccumBounds(-1, 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet:
                return AccumBounds(Min(sin(min), sin(max)), 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2), pi*Rational(8, 2))) \
                        is not S.EmptySet:
                return AccumBounds(-1, Max(sin(min), sin(max)))
            else:
                return AccumBounds(Min(sin(min), sin(max)),
                                Max(sin(min), sin(max)))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import sinh
            return S.ImaginaryUnit*sinh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.NegativeOne**(pi_coeff - S.Half)

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # https://github.com/sympy/sympy/issues/6048
            # transform a sine to a cosine, to avoid redundant code
            if pi_coeff.is_Rational:
                x = pi_coeff % 2
                if x > 1:
                    return -cls((x % 1)*pi)
                if 2*x > 1:
                    return cls((1 - x)*pi)
                narg = ((pi_coeff + Rational(3, 2)) % 2)*pi
                result = cos(narg)
                if not isinstance(result, cos):
                    return result
                if pi_coeff*pi != arg:
                    return cls(pi_coeff*pi)
                return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return sin(m)*cos(x) + cos(m)*sin(x)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, asin):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return x/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return y/sqrt(x**2 + y**2)

        if isinstance(arg, acos):
            x = arg.args[0]
            return sqrt(1 - x**2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/(sqrt(1 + 1/x**2)*x)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, asec):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        return (exp(arg*I) - exp(-arg*I))/(2*I)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*x**-I/2 - I*x**I /2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)
        return Piecewise((0, And(Eq(im(arg), 0), Eq(Mod(arg, pi), 0))),
                         (2*cot_half/(1 + cot_half**2), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/csc(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_sinc(self, arg, **kwargs):
        return arg*sinc(arg)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(pi*arg/2)*besselj(S.Half, arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (sin(re)*cosh(im), cos(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt, chebyshevu
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO, implement more if deep stuff here
            # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return sx*cy + sy*cx
        elif arg.is_Mul:
            n, x = arg.as_coeff_Mul(rational=True)
            if n.is_Integer:  # n will be positive because of .eval
                # canonicalization

                # See https://mathworld.wolfram.com/Multiple-AngleFormulas.html
                if n.is_odd:
                    return S.NegativeOne**((n - 1)/2)*chebyshevt(n, sin(x))
                else:
                    return expand_mul(S.NegativeOne**(n/2 - 1)*cos(x)*
                                      chebyshevu(n - 1, sin(x)), deep=False)
        return sin(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return pi_mult.is_integer

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
                or self.args[0].is_complex:
            return True


class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sympy.functions.elementary.trigonometric.sin
    sympy.functions.elementary.trigonometric.csc
    sympy.functions.elementary.trigonometric.sec
    sympy.functions.elementary.trigonometric.tan
    sympy.functions.elementary.trigonometric.cot
    sympy.functions.elementary.trigonometric.asin
    sympy.functions.elementary.trigonometric.acsc
    sympy.functions.elementary.trigonometric.acos
    sympy.functions.elementary.trigonometric.asec
    sympy.functions.elementary.trigonometric.atan
    sympy.functions.elementary.trigonometric.acot
    sympy.functions.elementary.trigonometric.atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cos

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_derivative_n_times(self, x, n):
        return cos(self.args[0] + n*S.Pi/2) * self.args[0].diff(x)**n

    @classmethod
    def eval(cls, arg):
        from sympy.functions.special.polynomials import chebyshevt
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg in (S.Infinity, S.NegativeInfinity):
                # In this case it is better to return AccumBounds(-1, 1)
                # rather than returning S.NaN, since AccumBounds(-1, 1)
                # preserves the information that sin(oo) is between
                # -1 and 1, where S.NaN does not do that.
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            return sin(arg + pi/2)
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_extended_real and arg.is_finite is False:
            return AccumBounds(-1, 1)

        if arg.could_extract_minus_sign():
            return cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import cosh
            return cosh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return (S.NegativeOne)**pi_coeff

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # cosine formula #####################
            # https://github.com/sympy/sympy/issues/6048
            # explicit calculations are performed for
            # cos(k pi/n) for n = 8,10,12,15,20,24,30,40,60,120
            # Some other exact values like cos(k pi/240) can be
            # calculated using a partial-fraction decomposition
            # by calling cos( X ).rewrite(sqrt)
            if pi_coeff.is_Rational:
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    return -cls(narg)

                # If nested sqrt's are worse than un-evaluation
                # you can require q to be in (1, 2, 3, 4, 6, 12)
                # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
                # expressions with 2 or fewer sqrt nestings.
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    a, b = p*pi/a, p*pi/b
                    nvala, nvalb = cls(a), cls(b)
                    if None in (nvala, nvalb):
                        return None
                    return nvala*nvalb + cls(pi/2 - a)*cls(pi/2 - b)

                if q > 12:
                    return None

                cst_table_some = {
                    3: S.Half,
                    5: (sqrt(5) + 1) / 4,
                }
                if q in cst_table_some:
                    cts = cst_table_some[pi_coeff.q]
                    return chebyshevt(pi_coeff.p, cts).expand()

                if 0 == q % 2:
                    narg = (pi_coeff*2)*pi
                    nval = cls(narg)
                    if None == nval:  # noqa: E711
                        return None
                    x = (2*pi_coeff + 1)/2
                    sign_cos = (-1)**((-1 if x < 0 else 1)*int(abs(x)))
                    return sign_cos*sqrt( (1 + nval)/2 )
            return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return cos(m)*cos(x) - sin(m)*sin(x)

        if arg.is_zero:
            return S.One

        if isinstance(arg, acos):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return 1/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return x/sqrt(x**2 + y**2)

        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x ** 2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/sqrt(1 + 1/x**2)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

        if isinstance(arg, asec):
            x = arg.args[0]
            return 1/x

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        return (exp(arg*I) + exp(-arg*I))/2

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return x**I/2 + x**-I/2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return sin(arg + pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)**2
        return (1 - tan_half)/(1 + tan_half)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/sin(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)**2
        return Piecewise((1, And(Eq(im(arg), 0), Eq(Mod(arg, 2*pi), 0))),
                         ((cot_half - 1)/(cot_half + 1), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._eval_rewrite_as_sqrt(arg, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg: Expr, **kwargs):
        from sympy.functions.special.polynomials import chebyshevt

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is None:
            return None

        if isinstance(pi_coeff, Integer):
            return None

        if not isinstance(pi_coeff, Rational):
            return None

        cst_table_some = cos_table()

        if pi_coeff.q in cst_table_some:
            rv = chebyshevt(pi_coeff.p, cst_table_some[pi_coeff.q]())
            if pi_coeff.q < 257:
                rv = rv.expand()
            return rv

        if not pi_coeff.q % 2:  # recursively remove factors of 2
            pico2 = pi_coeff * 2
            nval = cos(pico2 * pi).rewrite(sqrt, **kwargs)
            x = (pico2 + 1) / 2
            sign_cos = -1 if int(x) % 2 else 1
            return sign_cos * sqrt((1 + nval) / 2)

        FC = fermat_coords(pi_coeff.q)
        if FC:
            denoms = FC
        else:
            denoms = [b**e for b, e in factorint(pi_coeff.q).items()]

        apart = ipartfrac(*denoms)
        decomp = (pi_coeff.p * Rational(n, d) for n, d in zip(apart, denoms))
        X = [(x[1], x[0]*pi) for x in zip(decomp, numbered_symbols('z'))]
        pcls = cos(sum(x[0] for x in X))._eval_expand_trig().subs(X)

        if not FC or len(FC) == 1:
            return pcls
        return pcls.rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/sec(arg).rewrite(csc, **kwargs)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (sqrt(pi*arg/2)*besselj(-S.Half, arg), Ne(arg, 0)),
                (1, True)
            )

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (cos(re)*cosh(im), -sin(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return cx*cy - sx*sy
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer:
                return chebyshevt(coeff, cos(terms))
        return cos(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi/2)/pi
        if n.is_integer:
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]

        if arg.is_extended_real:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
            or self.args[0].is_complex:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero and pi_mult:
            return (pi_mult - S.Half).is_integer