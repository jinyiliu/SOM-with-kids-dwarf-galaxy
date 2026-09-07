import numpy as np
from astropy.cosmology import Planck18 as astropy_Planck18
from typing import Callable


h = astropy_Planck18.H0.value / 100


class SHMR:
    """Stellar-to-Halo Mass Ratio.

    Notes:
        Masses are in units of solar masses. No h scaling is used.
    """
    pass


class VanUitert2016(SHMR):
    log10_M0 = 10.58 - 2 * np.log10(h)
    log10_M1 = 10.97 - 1 * np.log10(h)
    beta1 = 7.5
    beta2 = 0.25
    label = "van Uitert et al. (2016)"
    data = "KiDS + GAMA"

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        M = 10 ** log10_M
        M1 = 10 ** cls.log10_M1
        log10_M_star = (
                cls.log10_M0 + cls.beta1 * np.log10(M/M1) -
                (cls.beta1 - cls.beta2) * np.log10(1 + (M/M1))
        )
        return log10_M_star - log10_M


class Dvornik2023(SHMR):
    log10_M0 = 10.521 - 2 * np.log10(h)
    log10_M1 = 11.145 - 1 * np.log10(h)
    gamma1 = 7.385
    gamma2 = 0.201
    label = "Dvornik et al. (2023)"
    data = "KiDS + GAMA"

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        M = 10 ** log10_M
        M1 = 10 ** cls.log10_M1
        log10_M_star = (
                cls.log10_M0 + cls.gamma1 * np.log10(M/M1) -
                (cls.gamma1 - cls.gamma2) * np.log10(1 + (M/M1))
        )
        return log10_M_star - log10_M


class Dvornik2020(SHMR):
    alpha = 12.
    gamma = 10.08
    beta = -2.95
    label = "Dvornik et al. (2020)"
    data = "KiDS + GAMA"

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        log10_M_star = cls.alpha - 10 ** (
            log10_M / cls.beta + cls.gamma / np.log(10)
        )
        return log10_M_star - log10_M