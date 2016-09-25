from particlegenerators import base
import numpy as np
from numpy import ndarray, cos, sin, exp, arccos
from functools import reduce
import logging

logging.basicConfig(level=0)

# electron charge (SI system)
e = 1.61 * 10 ** -19 * 9.486833 * 10 ** 19
# Bohr radius (SI system)
a0 = 5.29 * 10 ** -11


class Simulation:
    POTENTIAL = dict(moliere=[0.67430, 0.00961, 0.00518, 10.0, 6.31400],
                     VHB=[0.69905, 0.05534, 0.03326, 10.79853, 6.55075],
                     universal=[0.75984, 5.71974, 6.14171, 9.5217, 6.26120])

    SCREEN = dict(moliere=[[0.35, 0.3],
                           [0.55, 1.2],
                           [0.1, 6.0]],
                  VHB=[[0.0069, 0.132],
                       [0.167, 0.302],
                       [0.826, 0.917]],
                  universal=[[0.1818, 3.2],
                             [0.5099, 0.9423],
                             [0.2802, 0.4029],
                             [0.02817, 0.2016]])

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
            logging.warning("Invalid potential name. "
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
        rm = self.rm(energy, p)
        rho = self.rho(energy, rm)
        delta = self.delta(rm, p, energy)

        return arccos((rho + delta + rm) / (rho + rm))

    def rm(self, energy, impact_pars, eps=0.001):
        def func(r, impact):
            return 1 - self.potential(r) / energy - (impact / r) ** 2

        def func_deriv(r, impact):
            return -self.potential_deriv(r) / energy \
                   + 2 * impact ** 2 / r ** 3

        def solve(r, impact):
            logging.info('Solve running.')
            diff = func(r, impact) / func_deriv(r, impact)
            selection = np.abs(diff) > eps  # type: ndarray
            if selection.any():
                r[selection] = solve(r[selection] - diff[selection],
                                     impact[selection])
            else:
                r = r - diff

            return r

        return solve(np.copy(impact_pars), impact_pars)

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
        return reduce(lambda a, x: exp(-x[1] * r) * x[0] + a, pars,
                      np.zeros(len(r)))

    def screen_func_deriv(self, r: ndarray) -> ndarray:
        pars = self.SCREEN[self.potential_name]
        return reduce(lambda a, x: a - x[0] * x[1] * exp(-x[1] * r), pars,
                      np.zeros(len(r)))

    def potential(self, r: ndarray) -> ndarray:
        z1, z2 = self.z1, self.z2
        result = z1 * z2 * e ** 2 / r
        screen = self.screen_func(r)
        return result * screen

    def potential_deriv(self, r: ndarray) -> ndarray:
        return self.potential(r) / r * self.screen_func(r) \
               + self.potential(r) * self.screen_func_deriv(r)


if __name__ == '__main__':
    logging.info("Test run.")
    sim = Simulation()
    logging.info("Class initialization successful.")
    logging.info("Running rm method.")
    print(sim.rm(1000, np.arange(1, 10, 1) / 2.))
