"""
Module for calculating scattering angles in nuclei-nuclei
elastic collisions.

Calculates scattering angles based on energy and impact parameters.
Different potential forms are available:
    -- basic coulomb
    -- momentum approximation
    -- semi-empirical with different constant sets:
        -- Moliere
        -- VHB
        -- universal
"""

import numpy as np
from numpy import ndarray, exp, arccos, arcsin, sqrt
from functools import reduce
import logging

logging.basicConfig(level=logging.INFO)

e = 1.61 * 10 ** -19 * 9.486833 * 10 ** 14
"effective electron charge (SI system)"

a0 = 5.29 * 10 ** -1
"Bohr radius (SI system)"


class ScatteringElastic:
    e = 0.12
    "electron charge (keV/angstrem)"
    
    a0 = 0.529
    "Bohr radius (angstrem)"
    
    POTENTIAL = {'moliere'  : [0.67430, 0.00961, 0.00518, 10.0, 6.31400],
                 'VHB'      : [0.69905, 0.05534, 0.03326, 10.79853, 6.55075],
                 'universal': [0.75984, 5.71974, 6.14171, 9.5217, 6.26120]}
    "Parameters for approximation of potential."
    
    SCREEN = {'moliere'  : [[0.35, 0.3],
                            [0.55, 1.2],
                            [0.1, 6.0]],
              'VHB'      : [[0.0069, 0.132],
                            [0.167, 0.302],
                            [0.826, 0.917]],
              'universal': [[0.1818, 3.2],
                            [0.5099, 0.9423],
                            [0.2802, 0.4029],
                            [0.02817, 0.2016]]}
    "Parameters for calculating screening function."
    
    POTENTIAL_LIST_SEMIEMPIRIC = ['moliere', 'VHB', 'universal']
    
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
            logging.info("Potential changed to %s." % name)
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
    
    def calc_theta(self, energy: ndarray, impact_par: ndarray) -> ndarray:
        if self.potential_name in self.POTENTIAL_LIST_SEMIEMPIRIC:
            return self.calc_theta_semiempirical(energy, impact_par)
        elif self.potential_name == 'coulomb':
            return self.calc_coulomb(energy, impact_par)
        elif self.potential_name == 'momentum':
            return self.calc_momentum_approximation(energy, impact_par)
    
    def calc_theta(self, energy, impact_par):
        pass
    
    def calc_coulomb(self, energy: ndarray, impact_par: ndarray) -> ndarray:
        b = impact_par / self.a0
        "dimensionless impact parameter"
        en_lin = energy * self.a0 / (self.z1 * self.z2 * self.e ** 2)
        "lindhard energy"
        if en_lin.shape != b.shape:
            en_lin = np.array([v for (u, v) in np.broadcast(b, en_lin)])
        
        theta = sqrt(1. / (1 + (2 * en_lin * b) ** 2))
        
        return theta
    
    def calc_momentum_approximation(self, energy: ndarray,
                                    impact_par: ndarray) -> ndarray:
        a = 0.88534 * a0 / (self.z1 ** 0.23 + self.z2 ** 0.23)
        b = impact_par / a
        "dimensionless impact parameter"
        en_lin = energy * a / (self.z1 * self.z2 * self.e ** 2)
        "lindhard energy"
        if en_lin.shape != b.shape:
            en_lin = np.array([v for (u, v) in np.broadcast(b, en_lin)])
        
        theta = np.empty(impact_par.shape)
        
        selection = b <= 1.
        theta[selection] = 1.012 / b[selection] * exp(-0.356 * b[selection])
        
        selection = (1. < b) * (b <= 2.)
        theta[selection] = 0.707 * np.power(b[selection], -1.78)
        
        selection = 2. < b
        theta[selection] = 0.05 * exp(-0.308 * b[selection]) \
                           + 1.306 * exp(-1.012 * b[selection])
        
        return 1 / en_lin * theta, b
    
    def calc_theta_semiempirical(self, energy: ndarray,
                                 impact_par: ndarray) -> ndarray:
        """
        Main method of class. Calculate scattering angle for impact parameters
        and energy_test. Using currently selected potential for class.

        :param energy: vector or single value of energy_test
        :param impact_par: vector of impact parameters
        :return:
        """

        b = impact_par / self.a0
        "dimensionless impact parameter"
        en_lin = energy * self.a0 / (self.z1 * self.z2 * self.e)
        "lindhard energy"
        if en_lin.shape != b.shape:
            en_lin = np.array([v for (u, v) in np.broadcast(b, en_lin)])
        
        rm = self.rm(energy, imp_pars) / self.a0
        # print('rm\n', rm)
        rho = self.rho(energy, rm * self.a0) / self.a0
        # print(rho)
        delta = self.delta(rm, b, en_lin)
        # print(delta)
        
        temp = (b + delta + rho) / (rho + rm)
        
        return 2 * arccos(temp), b
    
    def rm(self, energy, impact_pars, eps=0.001):
        def func(r, energy_, impact):
            return 1 - self.potential(r) / energy_ - (impact / r) ** 2
        
        def func_deriv(r, energy_, impact):
            return - self.potential_deriv(r) / energy_ \
                   + 2 * impact ** 2 / r ** 3
        
        def solve(r, energy_, impact):
            logging.info('Solve running.')
            diff = func(r, energy_, impact) / func_deriv(r, energy_, impact)
            selection = np.abs(diff) > eps  # type: ndarray
            if selection.any():
                r[selection] = solve(r[selection] - diff[selection],
                                     energy_[selection], impact[selection])
            else:
                r = r - diff
            
            return r
        
        def solve_loop(energy_, impact):
            x0, x1 = impact.copy(), impact.copy()
            for i in range(100):
                x0 = x1
                x1 = x0 - func(x0, energy_, impact) \
                          / func_deriv(x0, energy_, impact)
            
            return x1

        logging.info("Running rm method.")
        
        return solve_loop(energy, impact_pars)
    
    def rho(self, energy, rm):
        return 2. * (self.potential(rm) - energy) / self.potential_deriv(rm)
    
    def delta(self, min_radius: ndarray, impact_parameter: ndarray,
              energy: ndarray) -> ndarray:
        pars = self.POTENTIAL[self.potential_name]
        
        alpha = 1 + pars[0] * energy ** -0.5
        beta = (pars[1] + energy ** 0.5) / (pars[2] + energy ** 0.5)
        gamma = (pars[3] + energy) / (pars[4] + energy)
        
        a = 2 * alpha * energy * impact_parameter ** beta
        g = gamma / ((1 + a ** 2) ** 0.5 - a)
        
        return a / (1 + g) * (min_radius - impact_parameter)
    
    def coulomb(self, r) -> ndarray:
        """
        Calculate basic coulomb potential.

        :param ndarray r: distances
        :return:
        """
        
        return self.e ** 2 * self.z1 * self.z2 / r
    
    def screen_func(self, r) -> ndarray:
        """
        Screening func for coulomb potential. Current potential approximation
        is used.

        :param ndarray r:
        :return: ndarray
        """
        
        pars = self.SCREEN[self.potential_name]
        a = 0.88534 * a0 / (self.z1 ** 0.23 + self.z2 ** 0.23)
        r_ = r / a
        return reduce(lambda acc, x: exp(-x[1] * r_) * x[0] + acc, pars,
                      np.zeros(len(r)))
    
    def screen_func_deriv(self, r) -> ndarray:
        """
        Derivative of screening func for coulomb potential.
        Current potential approximation is used.

        :param ndarray r:
        :return:
        """
        
        pars = self.SCREEN[self.potential_name]
        a = 0.88534 * a0 / (self.z1 ** 0.23 + self.z2 ** 0.23)
        r_ = r / a
        return reduce(lambda acc, x: acc - x[0] * x[1] * exp(-x[1] * r_), pars,
                      np.zeros(len(r)))
    
    def potential(self, r) -> ndarray:
        """
        Calculate coulomb potential with screening defined by current
        potential approximation.

        :param ndarray r:
        :return:
        """
        
        return self.coulomb(r) * self.screen_func(r)
    
    def potential_deriv(self, r) -> ndarray:
        """
        Calculate derivative of coulomb potential with screening
        defined by current potential approximation.

        :param ndarray r:
        :return:
        """
        
        return - self.coulomb(r) / r * self.screen_func(r) \
               + self.coulomb(r) * self.screen_func_deriv(r)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    logging.info("Test run.")
    sim = ScatteringElastic(z1=6, z2=18)
    logging.info("Class initialization successful.")
    # imp_pars = base.uniform(0.2, 10, 100) * a0 / 10.
    imp_pars = np.linspace(0.1, 1.5, 100)
    energy_test = np.array(100.)
    
    theta_coul = sim.calc_coulomb(energy_test, imp_pars)
    plt.plot(imp_pars, theta_coul, 'g.', label='Coulomb')

    theta = sim.calc_momentum_approximation(energy_test, imp_pars)[0]
    plt.plot(imp_pars, theta, 'r', label='Momentum')

    theta_semi = sim.calc_theta_semiempirical(energy_test, imp_pars)[0]
    print(theta_semi)
    plt.plot(imp_pars, theta_semi, 'b', label='Semiempirical')
    
    # plt.plot(imp_pars, sim.potential(imp_pars), 'b', label='potential')
    # plt.plot(imp_pars, sim.potential_deriv(imp_pars), 'r', label='deriv')
    
    # plt.yscale('log')
    plt.legend()
    plt.show()
