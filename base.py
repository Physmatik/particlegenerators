"""
Basic functions for particlegenerators package.
"""

import numpy as np
from numpy import cos, sin, arccos, arctan2, pi, ndarray


def uniform(a=0., b=1., size=1000) -> ndarray:
    """
    Return numpy ndarray with uniformly
    distributed random numbers from a to b.

    :param int, tuple size: number of values to generate
    :param float a: minimum value
    :param float b: maximum value
    """

    return np.random.random(size) * (b - a) + a


def gauss(mu=0., sigma=1.0, size: int = 1000, quality: int = 12) \
        -> ndarray:
    """
    Return numpy array with gauss distributed values.

    Generation is done by summing multiple uniform distributions
    and some extended transformations.

    :param float mu: mean value of gauss distribution.
    :param float sigma: standard deviation of gauss distribution.
    :param int size: number of values in return array.
    :param int quality: number of distibutions to sum (the higher the better).
    """

    ksi = np.sum(np.random.random((quality, size)) - 0.5,
                 axis=0) * 12 / quality ** 0.5

    eta = (ksi + (ksi * ksi * ksi - ksi * 3) / (20 * quality)) * sigma + mu
    return eta


def cosines_to_angles(cosines) -> ndarray:
    """
    Convert direction cosines to spherical angles.

    :param (3, N) ndarray cosines: rows must be cosines
    :return: (2, N) ndarray. rows are (theta, phi)
    """

    x, y, z = cosines[0], cosines[1], cosines[2]
    theta = arccos(z)
    phi = arctan2(y, x)

    return np.vstack((theta, phi))


def angles_to_cosines(angles) -> ndarray:
    """
    Convert spherical angles to direction cosines.

    :param (2, N) ndarray angles: rows are theta, phi
    :return: (3, N) ndarray. array of cosines
    """

    theta, phi = angles[0], angles[1]

    cos_x = sin(theta) * cos(phi)
    cos_y = sin(theta) * sin(phi)
    cos_z = cos(theta)

    return np.vstack((cos_x,
                      cos_y,
                      cos_z))


def rotation_matrix(axis='z', angle=pi / 2.) -> ndarray:
    """
    Return rotation matrix in 3D-space.

    :param char axis: axis to rotate around
    :param float angle: angle of rotation
    :return:
    """

    c, s = cos(angle), sin(angle)
    if axis == 'z':
        matrix = np.array([[c, -s,  0.],
                           [s,  c,  0.],
                           [0., 0., 1.]])
    elif axis == 'y':
        matrix = np.array([[c,  0., -s],
                           [0., 1., 0.],
                           [s,  0.,  c]])
    elif axis == 'x':
        matrix = np.array([[1., 0., 0.],
                           [0.,  c, -s],
                           [0.,  s,  c]])
    else:
        raise ValueError("Axis must 'x', 'y' or 'z'.")

    return matrix
