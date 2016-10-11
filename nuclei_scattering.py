"""
Module for calculating scattering angles in nuclei-nuclei collisions.

Calculates scattering angles based on energy and impact parameters.
Different potential forms are available:
    -- basic coulomb
    -- momentum approximation
    -- semi-empirical with different constant sets:
        -- Moliere
        -- VHB
        -- universal
"""

from particlegenerators import base
import numpy as np
from numpy import ndarray, exp, arccos
from functools import reduce
import logging

logging.basicConfig(level=logging.WARN)

e = 1.61 * 10 ** -19 * 9.486833 * 10 ** 14
"effective electron charge (SI system)"

a0 = 5.29 * 10 ** -11
"Bohr radius (SI system)"


# noinspection PyShadowingNames
class Simulation:
    POTENTIAL = dict(moliere=[0.67430, 0.00961, 0.00518, 10.0, 6.31400],
                     VHB=[0.69905, 0.05534, 0.03326, 10.79853, 6.55075],
                     universal=[0.75984, 5.71974, 6.14171, 9.5217, 6.26120])
    "Parameters for approximation of potential."

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
    "Parameters for calculating screening function."

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

    def set_potential(self, name: str) -> None:
        """
        Change current potential of simulation.

        :param str name: name of potential or it's approximation parameters set
        """
        if name.lower() in ['moliere', 'VHB', 'universal']:
            self.potential_name = name
            logging.info("Potential changed to {}.".format(name))
        else:
            logging.warning("Invalid potential name. "
                            "Current potential unchanged.")

    def set_charge(self, z1=None, z2=None) -> None:
        """
        Change current values for charge of target and bombarding particles.

        :param int z1: charge of target in electron charge units
        :param itn z2: charge of bombarding particles in electron charge units
        """

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

    def calculate_theta(self, energy: ndarray, impact_par: ndarray) -> ndarray:
        """
        Main method of class. Calculate scattering angle for impact parameters
        and energy. Using currently selected potential for class.

        :param Union[number, ndarray] energy:
        :param Union[number, ndarray] impact_par:
        :return:
        """

        rm = self.rm(energy, impact_par)
        rho = self.rho(energy, rm)
        delta = self.delta(rm, impact_par, energy)

        temp = a0 * (impact_par + delta + rho) / (rho + rm)
        print(temp)
        selection = (temp < 1.) * (temp > -1.)

        return 2 * arccos(temp[selection]), impact_par[selection]

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

    def screen_func(self, r) -> ndarray:
        """
        Screening func for coulomb potential. Current potential approximation
        is used.

        :param ndarray r:
        :return: ndarray
        """
        pars = self.SCREEN[self.potential_name]
        return reduce(lambda a, x: exp(-x[1] * r) * x[0] + a, pars,
                      np.zeros(len(r)))

    def screen_func_deriv(self, r) -> ndarray:
        """
        Derivative of screening func for coulomb potential.
        Current potential approximation is used.

        :param ndarray r:
        :return:
        """

        pars = self.SCREEN[self.potential_name]
        return reduce(lambda a, x: a - x[0] * x[1] * exp(-x[1] * r), pars,
                      np.zeros(len(r)))

    def potential(self, r) -> ndarray:
        """
        Calculate basic coulomb potential.

        :param ndarray r:
        :return:
        """

        z1, z2 = self.z1, self.z2
        return z1 * z2 * e ** 2 / r * self.screen_func(r)

    def potential_deriv(self, r) -> ndarray:
        """
        Calculate derivative of basic coulomb potential.

        :param ndarray r:
        :return:
        """
        return self.potential(r) / r * self.screen_func(r) \
               + self.potential(r) * self.screen_func_deriv(r)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.info("Test run.")
    sim = Simulation()
    logging.info("Class initialization successful.")
    logging.info("Running rm method.")
    impact_par = base.uniform(0.2, 10, 100) * a0 / 10.
    impact_par = np.arange(5, 100) / 200 * a0
    energy = 1
    theta, impact_par = sim.calculate_theta(energy, impact_par)
    theta = (theta - np.pi) * 180 / np.pi

    plt.plot(impact_par, theta, 'k-')
    plt.show()
