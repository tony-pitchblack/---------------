from scipy import stats, special
from scipy.optimize import fsolve
import numpy as np

def sample_size_rep(alpha, delta, sigma=None, std=None):
    # repetitive
    if sigma and not std:
        # sigma known
        def laplace_eq(u_crit):
            """https://stackoverflow.com/questions/56016484/how-to-calculate-laplace-function-in-python-3"""
            Phi = lambda x: special.erf(x/2**0.5)/2
            f = -Phi(u_crit) + ((1-alpha)/2)
            return f
        u_crit = fsolve(laplace_eq, 2)[0]
        n = pow(u_crit, 2) * pow(sigma, 2) / pow(delta, 2)
        return n
    elif std and not sigma:
        # sigma unknown
        def sample_size_equation(n):
            t_crit = stats.t.ppf(1-alpha, n-1)
            f = -n + pow(t_crit, 2) * pow(std, 2) / pow(delta, 2)
            return f
        n = fsolve(sample_size_equation,2)[0]
        return n
    else:
        raise ValueError("Exactly one of sigma and std has to be set, and the other needs to be None.")

def sample_size(alpha, delta, N, sigma=None, std=None):
        if std and not sigma:
            # sigma unknown
            def sample_size_equation(n):
                t_crit = stats.t.ppf(1-alpha, n-1)
                f = -n + pow(t_crit, 2) * pow(std, 2) * N / \
                    ( pow(delta, 2) * N + pow(t_crit, 2) * pow(std, 2) )
                return f
            n = fsolve(sample_size_equation, 2)[0]
            return n
        else:
            raise ValueError

