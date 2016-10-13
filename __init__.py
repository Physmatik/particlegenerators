"""
Set of tools for generating different distributions.

Currently present:
    * base:
        * uniform [a, b)
        * gauss(mu, sigma)
    * directions:
        * isotropic (in 4pi)
        * cone (specified body angle)
    * coordinates:
        * flat disk
        * orb
        * cylinder
        * empty orb
    * nuclei scattering:
        * scattering angles for different potentials
"""

import logging

from .base import *
from . import coordinates
from . import directions
from . import nuclei_scattering

__all__ = ['base', 'coordinates', 'directions', 'nuclei_scattering']

logging.info('particlegenerators module initialized.')
