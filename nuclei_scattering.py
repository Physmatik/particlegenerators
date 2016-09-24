from particlegenerators import base
import numpy as np
from numpy import ndarray, cos, sin, exp
from functools import reduce
import logging

# electron charge (SI system)
e = 1.61 * 10 ** -19
# Bohr radius (SI system)
a0 = 5.29 * 10 ** -11


class Simulation:
    POTENTIAL = {
        'moliere': [0.67430, 0.00961, 0.00518, 10.0, 6.31400],
        'VHB': [0.69905, 0.05534, 0.03326, 10.79853, 6.55075],
        'universal': [0.75984, 5.71974, 6.14171, 9.5217, 6.26120]
    }

    SCREEN = {
        'moliere': [[0.35, 0.55, 0.1],
                    [0.3, 1.2, 6]],
        'VHB': [[0.0069, 0.167, 0.826],
                [0.132, 0.302, 0.917]],
        'universal': [[0.1818, 0.5099, 0.2802, 0.02817],
                      [3.2, 0.9423, 0.4029, 0.2016]]
    }

    def __init__(self, potential_type='universal', z1=1, z2=1, m1=1, m2=1):
        if potential_type in ['moliere', 'VHB', 'universal']:
            self.potential_name = potential_type
        else:
            self.potential_name = 'universal'
            logging.warning("Invalid potential name. "
                            "Set to universal by default.")
        self.z1 = z1
        self.z2 = z2
        self.m1 = m1
        self.m2 = m2

    def set_potential(self, s: str) -> None:
        if s.lower() in ['moliere', 'VHB', 'universal']:
            self.potential_name = s
            logging.info("Potential changed to {}.".format(s))
        else:
            logging.warning("Unknown potential name. "
                            "Current potential unchanged.")

    def set_charge(self, z1=None, z2=None):
        try:
            if float(z1).is_integer():
                self.z1 = z1
            else:
                logging.warning(
                    "Non-integer passed as charge. State wasn't changed.")
            if float(z2).is_integer():
                self.z2 = z2
            else:
                logging.warning(
                    "Non-integer passed as charge. State wasn't changed.")
        except ValueError:
            logging.warning("Unexpected exception while setting charge."
                            " Check input parameters.")

    def calculate_theta(self, energy: ndarray, p: ndarray) -> ndarray:
        pass

    def rm(self, energy, impact_pars, eps=0.001):
        def func(r):
            return 1 - self.potential(r) / energy - (impact_pars / r) ** 2

        def func_deriv(r):
            return -self.potential_deriv(r) / energy \
                   + 2 * impact_pars ** 2 / r ** 3

        def solve(r):
            diff = func(r) / func_deriv(r)
            selection = diff > eps
            if selection.any():
                r[selection] = solve(r[selection] - diff[selection])
            else:
                return r

        return solve(np.copy(impact_pars))

    def rho(self, energy, rm):
        return 2 * (self.potential(rm) - energy) / self.potential_deriv(rm)

    def delta(self, min_radius: ndarray, impact_parameter: ndarray,
              energy: ndarray) -> ndarray:
        pars = self.POTENTIAL[self.potential_name]

        alpha = 1 + pars[0] * energy ** -0.5
        beta = (pars[1] + energy ** 0.5) / (pars[2] + energy ** 0.5)
        gamma = (pars[3] + energy) / (pars[4] + energy)

        a = 2 * alpha * energy * impact_parameter ** beta
        g = gamma / ((1 + a ** 2) ** 0.5 - a)

        return a / (1 + g) * (min_radius - impact_parameter)

    def screen_func(self, r: ndarray) -> ndarray:
        pars = self.SCREEN[self.potential_name]
        return reduce(lambda x, a: a + x[0] * exp(-x[1] * r), pars)

    def screen_func_deriv(self, r: ndarray) -> ndarray:
        pars = self.SCREEN[self.potential_name]
        return reduce(lambda x, a: a - x[0] * x[1] * exp(-x[1] * r), pars)

    def potential(self, r: ndarray) -> ndarray:
        z1, z2 = self.z1, self.z2
        return z1 * z2 * e ** 2 / r * self.screen_func(r)

    def potential_deriv(self, r: ndarray) -> ndarray:
        return self.potential(r) / r * self.screen_func(r) \
               + self.potential(r) * self.screen_func_deriv(r)
