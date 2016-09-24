import numpy as np
from numpy import cos, sin, arccos, arctan2, pi


def uniform(a=0., b=1., size=1000) -> np.ndarray:
    """
    Return numpy ndarray with uniformly
    distributed random numbers from a to b.
    :param size: int, tuple
        number of values to generate. Can possibly be numpy tuple of sizes.
    :param a: float
        minimum value.
    :param b: float
        maximum value.
    """

    return np.random.random(size) * (b - a) + a


def gauss(mu=0., sigma=1.0, size: int = 1000, quality: int = 12) \
        -> np.ndarray:
    """
    Return numpy array with gauss distributed values.

    Generation is done by summing multiple uniform distributions
    and some extended transformations.
    :param mu: float
        mean value of gauss distribution.
    :param sigma: float
        standard deviation of gauss distribution.
    :param size: int
        number of values in return array.
    :param quality: int
        number of distibutions to sum (the higher the better).
    """

    ksi = np.sum(np.random.random((quality, size)) - 0.5,
                 axis=0) * 12 / quality ** 0.5

    eta = (ksi + (ksi * ksi * ksi - ksi * 3) / (20 * quality)) * sigma + mu
    return eta


def cosines_to_angles(cosines) -> np.ndarray:
    """
    Convert direction cosines to spherical angles.

    :param cosines: (3, N) ndarray
        rows must be cosines
    :return: (2, N) ndarray
        rows are (theta, phi)
    """

    x, y, z = cosines[0], cosines[1], cosines[2]
    theta = arccos(z)
    phi = arctan2(y, x)

    return np.vstack((theta, phi))


def angles_to_cosines(angles) -> np.ndarray:
    """
    Convert spherical angles to direction cosines.

    :param angles: (2, N) ndarray
        first row is theta, second is phi
    :return: (3, N) ndarray
        array of cosines
    """

    theta, phi = angles[0], angles[1]

    cos_x = sin(theta) * cos(phi)
    cos_y = sin(theta) * sin(phi)
    cos_z = cos(theta)

    return np.vstack((cos_x, cos_y, cos_z))


def rotation_matrix(axis='z', angle=pi / 3.) -> np.ndarray:
    """

    :param axis: char
        axis to rotate around
    :param angle: float
        angle of rotation
    :return:
    """

    c, s = cos(angle), sin(angle)
    if axis == 'z':
        matrix = np.array([[c, -s, 0.],
                           [s, c, 0.],
                           [0., 0., 1.]])
    elif axis == 'y':
        matrix = np.array([[c, 0., -s],
                           [0., 1., 0.],
                           [s, 0., c]])
    elif axis == 'x':
        matrix = np.array([[1., 0., 0.],
                           [0., c, -s],
                           [0., s, c]])
    else:
        raise ValueError("Axis must 'x', 'y' or 'z'.")

    return matrix


if __name__ == '__main__':
    print("Test run.")

    import matplotlib.pyplot as plt

    print("running generate_gauss()")
    plt.hist(gauss(size=10000), bins=100)

    print("gauss added to pyplot queue. showing at the end.")

    plt.show()
