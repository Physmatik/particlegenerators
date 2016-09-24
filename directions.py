import numpy as np
from numpy.core.multiarray import ndarray
from numpy.core.umath import cos, sin, pi

from particlegenerators import base
import matplotlib.pyplot as plt


def rotation_matrix(axis='z', angle=pi / 3.) -> ndarray:
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


def generate_isotropic(size=1000, cosines=True) -> ndarray:
    """
    Generate direction for isotropic source.

    :param size: int
        number of values to generate.
    :param cosines: boolean
        representation of values.
    :return: ndarray
        (3, size) ndarray if cosines==True. Those are direction cosines.
        (2, size) ndarray if cosines!=True. Those are angles (phi, theta).
    """

    phi = base.uniform(0, 2 * pi, size=size)
    theta = np.arccos(2. * base.uniform(size=size) - 1.)

    if cosines:
        print('Returned direction cosines.')
        return np.array([sin(theta) * cos(phi),
                         sin(theta) * sin(phi),
                         cos(theta)])
    else:
        print('Returned angles.')
        return np.vstack((phi, theta))


def generate_cone(phi_0=0., theta_0=pi / 4., d_theta=pi / 10.,
                  size=1000, cosines=True) -> ndarray:
    """
    Generate directions in form of cone.

    :param phi_0: float
        direction of the cone in the xy plane
    :param theta_0: float
        direction of the cone in the xz plane
    :param d_theta: float
        half-apperture of cone
    :param size: int
        number of directions to generate
    :param cosines: boolean
        representation of values
    :return: ndarray
        (3, size) ndarray if cosines==True. Those are direction cosines.
        (2, size) ndarray if cosines!=True. Those are angles (phi, theta).
    """

    theta = np.arccos(base.uniform(cos(d_theta), 1.0, size))
    phi = base.uniform(0, 2 * pi, size)

    if cosines:
        data = base.angles_to_cosines(np.array([theta, phi]))
        print('Return direction cosines.')
        return (base.rotation_matrix(axis='z', angle=phi_0)
                @ base.rotation_matrix(axis='y', angle=theta_0)
                @ data)

    else:
        print('Return angles.')
        return np.vstack((phi, theta))


def test_isotropic(size=1000):
    """
    Generate and display isotropic directions.

    Random lengths in range [0, 1) are used to produce filled orb.
    Pyplot.show() is not called.
    :param size: int
        number of points
    """

    data = generate_isotropic(size=size, cosines=False)
    data = np.vstack((base.uniform(size=size), data))

    data = np.vstack((data[0] * sin(data[2]) * cos(data[1]),
                      data[0] * sin(data[2]) * sin(data[1]),
                      data[0] * cos(data[2])))

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
        data = np.vstack((base.uniform(size=1000),
                          generate_cone(
                              theta_0=0.0,
                              d_theta=pi / 4.,
                              size=size,
                              cosines=False
                          )))

        data = np.vstack((data[0] * sin(data[2]) * cos(data[1]),
                          data[0] * sin(data[2]) * sin(data[1]),
                          data[0] * cos(data[2])))

        ax.plot(data[0], data[1], data[2], 'k.')
        ax.set_zlabel('Z')

    else:
        data = np.vstack((generate_cone(
            theta_0=pi / 8.,
            phi_0=pi / 4.,
            d_theta=pi / 4.,
            size=size,
            cosines=True
        )))

        r = base.uniform(size=size)

        data = data * r

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
