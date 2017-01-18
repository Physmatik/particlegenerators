import numpy as np
from numpy import sqrt, exp, power, pi, sin, ndarray
from base import uniform
from elastic_scattering import ScatteringElastic


def calc_impact_par(size, concent=6.02e23):
    return sqrt(uniform(size=size)/(pi * power(concent, 2. / 3.)))


def calc_partner(elements: list, size):
    indices = np.random.choice(len(elements[2]), p=elements[2], size=size)
    z2s = np.array(elements[0])[indices]
    m2s = np.array(elements[1])[indices]
    
    return z2s, m2s


def get_theta(energy: ndarray, impact_par, z1, m1, z2, m2):
    sim = ScatteringElastic(z1=z1, m1=m1, z2=z2, m2=m2)
    
    # energy to center of mass system
    energy_c = energy / (1 + m1 / m2)
    
    theta = sim.calc_momentum_approximation(energy_c, impact_par)
    
    return theta
    

def calc_energy_loss(energy: ndarray, theta, m1, m2):
    # energy transferred
    energy_t = energy * sin(theta / 2) ** 2
    energy_max = 4 * m1 * m2 * energy / (m1 + m2)
    energy_t[energy_t < energy_max] = energy_max
    
    # TODO energy in inelastic collisions
    
    return energy - energy_t


def step(coords, cosines, energy, z1, m1, target_elements):
    """

    :param coords:
    :param cosines:
    :param energy:
    :param run_len:
    :param z1:
    :param m1:
    :param target_elements: dictionary
        first line charge,
        second -- mass,
        third -- number of atoms in molecule
    """
    
    particle_num = len(energy)
    # get impact parameter
    impact_pars = calc_impact_par(particle_num)
    
    # get collision partner as an array of indices
    z2, m2 = calc_partner(target_elements, particle_num)
    
    theta = get_theta(energy, impact_pars, z1, m1, z2, m2)
    energy = calc_energy_loss(energy, theta, m1, m2)
