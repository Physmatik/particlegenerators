import numpy as np
import matplotlib.pyplot as plt

from particlegenerators import base


def disk(radius: float = 1.0, size: int = 1000,
         polar=False) -> np.ndarray:
    """
    Generate random position in disk source.

    Origin in (0,0,0).
    :param radius: float
        radius of source.
    :param size: int
        number of values to return
    :param polar: Boolean
        representation of coordinates
    :return: (3, size) numpy array.
        (rho, phi, 0) if polar True, (x, y, 0) otherwise.
    """

    # ----------------------------------------------------------------------
    # generate position
    rho = np.sqrt(base.uniform(size=size)) * radius
    phi = base.uniform(size=size) * 2 * np.pi
    z = np.zeros(size)

    if polar:
        position = np.vstack((rho,
                              phi,
                              z))
    else:
        position = np.vstack((rho * np.cos(phi),
                              rho * np.sin(phi),
                              z))

    return position


def orb(radius=1., size=1000, spherical=False) -> np.ndarray:
    """
    Generate random positions in orb source.

    Origin in (0,0,0).
    :param radius: float
        radius of sphere
    :param size: int
        number of values to generate
    :param spherical: Boolean
        representation of coordinates
    :return: (3, size) numpy array
        (rho, phi, theta) if spherical=True, (x, y, z) otherwise.
    """

    # noinspection PyTypeChecker
    rho = np.power(base.uniform(size=size), 1. / 3.) * radius
    phi = base.uniform(size=size) * 2 * np.pi
    theta = np.arccos(2 * base.uniform(size=size) - 1.)

    if spherical:
        position = np.vstack((rho,
                              phi,
                              theta))
    else:
        position = np.vstack((rho * np.sin(theta) * np.cos(phi),
                              rho * np.sin(theta) * np.sin(phi),
                              rho * np.cos(theta)))

    return position


def cylinder(radius=1.0, height=1.0, size=1000,
             cylindrical=False) -> np.ndarray:
    """
    Generate random positions in cylindrical source.

    Origin in (0,0,0)
    :param radius: float
        radius of cylinder
    :param height: float
        height of cylinder
    :param size: int
        number of values to generate
    :param cylindrical: Boolean
        representation of coordinates
    :return: (3, size) numpy array
        (rho, phi, z) if cylindrical=True, (x, y, z) otherwise.
    """
    if cylindrical:
        position = disk(radius=radius, size=size, polar=True)
    else:
        position = disk(radius=radius, size=size)
    position[2] = base.uniform(b=height, size=size)

    return position


def orb_gap(r1=0.5, r2=1.0, size=1000,
            spherical=False) -> np.ndarray:
    """
    Generate position for orb with hole inside.

    Origin in (0,0,0).
    :param r1: float
        inner radius
    :param r2: float
        outer radius
    :param size: int
        number of values to generate
    :param spherical: Boolean
        representation of coordinates
    :return: np.ndarray (3, size)
        (rho, phi, theta) if spherical is True, (x, y, z) otherwise.
    """

    # noinspection PyTypeChecker
    rho = np.power(
        base.uniform(size=size) * (r2 ** 3 - r1 ** 3) + r1 ** 3,
        1. / 3.)
    position = orb(radius=r2, size=size, spherical=True)
    position[0] = rho

    if spherical:
        return position
    else:
        return np.vstack((rho * np.sin(position[2]) * np.cos(position[1]),
                          rho * np.sin(position[2]) * np.sin(position[1]),
                          rho * np.cos(position[2])))


def test_disk():
    """
    Generate and display disk source.
    """
    fig_disk = plt.figure()
    fig_disk.suptitle('disk')
    ax_disk = fig_disk.add_subplot(111, projection='3d')
    data_disk = disk()
    ax_disk.plot(data_disk[0], data_disk[1], data_disk[2], 'k.')


def test_orb():
    """
    Generate and display orb source.
    """
    fig_orb = plt.figure()
    fig_orb.suptitle('orb')
    ax_orb = fig_orb.add_subplot(111, projection='3d')
    data_orb = orb(size=10000)
    ax_orb.plot(data_orb[0], data_orb[1], data_orb[2], 'k.')


def test_cylinder():
    """
    Generate and display cylinder source.
    """

    fig_cylinder = plt.figure()
    fig_cylinder.suptitle('cylinder')
    ax_cylinder = fig_cylinder.add_subplot(111, projection='3d')
    data_cylinder = cylinder(size=10000)
    ax_cylinder.plot(data_cylinder[0], data_cylinder[1],
                     data_cylinder[2], 'k.')


def test_orb_gap(cut=0.5, size=5000):
    """
    Generate and display source in form of orb with hole inside source.

    :param size: int
        number of points
    :param cut: float
        width of vertical slice (for better visual representation.
    """

    fig_orb_gap = plt.figure()
    fig_orb_gap.suptitle('two concentric orbs')
    ax_orb_gap = fig_orb_gap.add_subplot(111, projection='3d')

    data_orb_gap = orb_gap(size=size)

    selection = data_orb_gap[0] > -cut
    data_orb_gap = data_orb_gap[:, selection]

    selection = data_orb_gap[0] < cut
    data_orb_gap = data_orb_gap[:, selection]

    ax_orb_gap.plot(data_orb_gap[0], data_orb_gap[1],
                    data_orb_gap[2], 'k.')

    ax_orb_gap.set_xlim(-1., 1.)
    ax_orb_gap.set_ylim(-1., 1.)
    ax_orb_gap.set_zlim(-1., 1.)


def test_run():
    """Run simple test for class methods."""

    print("Test run.")

    test_disk()
    test_orb()
    test_cylinder()
    test_orb_gap()

    plt.show()


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    test_run()
