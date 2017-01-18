"""
Collection of functions that generate directions.

List of what is supported:
    * isotropic
    * cone

All function can return result both as direction cosines and spherical angles.
"""

import numpy as np
from numpy import cos, sin, pi, ndarray

from particlegenerators import base
import matplotlib.pyplot as plt


def generate_isotropic(size=1000, cosines=True) -> ndarray:
    """
    Generate direction for isotropic source.

    :param int size: number of values to generate
    :param boolean cosines: representation of values.
    :return: ndarray
        (3, size) ndarray if cosines==True. Those are direction cosines.
        (2, size) ndarray if cosines!=True. Those are angles (phi, theta).
    """
    
    phi = base.uniform(0, 2 * pi, size=size)
    theta = np.arccos(2. * base.uniform(size=size) - 1.)
    
    if cosines:
        return np.array([sin(theta) * cos(phi),
                         sin(theta) * sin(phi),
                         cos(theta)])
    else:
        return np.vstack((phi,
                          theta))


def generate_cone(phi_0=0., theta_0=pi / 4., d_theta=pi / 10.,
                  size=1000, cosines=True) -> ndarray:
    """
    Generate directions in form of cone.

    :param float phi_0: direction of the cone in the xy plane
    :param float theta_0: direction of the cone in the xz plane
    :param float d_theta: half-apperture of cone
    :param int size: number of directions to generate
    :param boolean cosines: representation of values
    :return: ndarray
        (3, size) ndarray if cosines==True. Those are direction cosines.
        (2, size) ndarray if cosines!=True. Those are angles (phi, theta).
    """
    
    theta = np.arccos(base.uniform(cos(d_theta), 1.0, size))
    phi = base.uniform(0, 2 * pi, size)
    
    if cosines:
        data = base.angles_to_cosines(np.array([theta, phi]))
        return (base.rotation_matrix(axis='z', angle=phi_0)
                @ base.rotation_matrix(axis='y', angle=theta_0)
                @ data)
    
    else:
        return np.vstack((phi,
                          theta))


def test_isotropic(size=1000):
    """
    Generate and display isotropic directions. Pyplot.show() is not called.
    
    :param int size: number of points
    """
    
    data = generate_isotropic(size=size, cosines=True)
    
    fig = plt.figure()
    fig.suptitle('isotropic')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], 'k.')


def test_cone(cosines=False, size=1000):
    """
    Generate and display conic directions.

    Random lengths in range [0, 1) are used to produce filled cone.
    Pyplot.show() is not called.
    :param cosines: boolean
        define whether direction cosines should be generated or angles
    :param size: int
        number of points
    """
    
    fig = plt.figure()
    fig.suptitle('distributions in form of cone')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_zlim(0., 1.)
    
    if not cosines:
        data = np.vstack((np.ones(size),
                          generate_cone(
                              theta_0=0.0,
                              d_theta=pi / 12.,
                              size=size,
                              cosines=False)))
        
        data = np.vstack((data[0] * sin(data[2]) * cos(data[1]),
                          data[0] * sin(data[2]) * sin(data[1]),
                          data[0] * cos(data[2])))
        
        ax.plot(data[0], data[1], data[2], 'k.')
        ax.set_zlabel('Z')
    
    else:
        data = np.vstack((generate_cone(theta_0=pi / 8., phi_0=pi / 4.,
                                        d_theta=pi / 40., size=size)))
        
        data = data
        
        ax.plot(data[0], data[1], data[2], 'k.')


def test_run():
    """
    Simple test run of class methods.
    """
    
    test_cone()
    test_cone(cosines=True)
    
    test_isotropic()
    
    plt.show()


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import mpl_toolkits.mplot3d.axes3d
    
    test_run()
