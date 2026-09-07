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


class Girelli2020(SHMR):
    B, mu = 11.79, 0.20
    C, nu = 0.046, -0.38
    D, eta = 0.709, -0.18
    F, E = 0.043, 0.96
    label = "Girelli et al. (2020)"
    data = "COSMOS + DUSTGRAIN"
    method = "AM"  # Abundance matching

    @classmethod
    def shmr(cls, log10_M: np.ndarray, z: float=0.1) -> np.ndarray:
        log10_MA = cls.B + cls.mu * z
        A = cls.C * (1.0 + z) ** cls.nu
        gamma = cls.D * (1.0 + z) ** cls.eta
        beta = cls.E + cls.F * z

        x = (10 ** np.asarray(log10_M, dtype=float)) / (10 ** log10_MA)
        ratio = 2.0 * A / (x ** (-beta) + x ** gamma)
        return np.log10(ratio)



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


class Hudson2015(SHMR):
    f0p5 = 0.034
    fz = 0.03
    log10_M0p5 = 12.5
    Mz = 0.4
    beta = 0.55
    gamma = 0.8
    label = "Hudson et al. (2015)"
    data = "CFHTLenS (Blue)"
    method = "GGL"

    @classmethod
    def shmr(cls, log10_M: np.ndarray, z: float=0.1) -> np.ndarray:
        f1 = cls.f0p5 + (z - 0.5) * cls.fz
        log10_M1 = cls.log10_M0p5 + (z - 0.5) * cls.Mz
        x = (10 ** np.asarray(log10_M, dtype=float)) / (10 ** log10_M1)
        return np.log10(2.0 * f1 / (x ** (-cls.beta) + x ** cls.gamma))


class Moster2010(SHMR):
    ratio_0 = 0.02820
    log10_M1 = 11.884
    beta = 1.057
    gamma = 0.556
    label = "Moster et al. (2010)"
    data = "SDSS + Millennium"
    method = "AM"  # Abundance matching

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        x = (10 ** np.asarray(log10_M, dtype=float)) / (10 ** cls.log10_M1)
        return np.log10(
            2.0 * cls.ratio_0 / (x ** (-cls.beta) + x ** cls.gamma)
        )


