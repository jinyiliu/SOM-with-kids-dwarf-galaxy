import os
import yaml
import numpy as np
from astropy.cosmology import Planck18 as astropy_Planck18
from typing import Callable
from scipy.interpolate import interp1d


h = astropy_Planck18.H0.value / 100


class SHMR:
    """Stellar-to-Halo Mass Ratio.

    Notes:
        Masses are in units of solar masses. No h scaling is used.
    """
    pass

class Chaikin2026(SHMR):
    label = "Chaikin+26"
    data = "COLIBRE"
    method = "hydro-sim"
    min_log10_M = 10.5
    _colibre_smhm = None

    @classmethod
    def shmr(cls, log10_M: np.ndarray, z: float=0.1) -> np.ndarray:
        with open(
            os.path.join(
                os.path.dirname(__file__), "data", "COLIBRE_SMHM.yaml"
            ),
            "r",
        ) as f:
            cls._colibre_smhm = yaml.load(f, Loader=yaml.SafeLoader)

        x = np.asarray(cls._colibre_smhm["m6"]["z0.1"]["x"], dtype=float)
        y = np.asarray(cls._colibre_smhm["m6"]["z0.1"]["y"], dtype=float)

        log10_ratio = interp1d(
            np.log10(x),
            np.log10(y),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return log10_ratio(np.asarray(log10_M, dtype=float))



class Girelli2020(SHMR):
    B, mu = 11.79, 0.20
    C, nu = 0.046, -0.38
    D, eta = 0.709, -0.18
    F, E = 0.043, 0.96
    label = "Girelli+20"
    data = "COSMOS + DUSTGRAIN"
    method = "AM"  # Abundance matching
    min_log10_M = 10.4

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
    label = "van Uitert+16"
    data = "KiDS-1000"
    min_log10_M = 11.0

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
    log10_M0 = 10.519 - 2 * np.log10(h)
    log10_M1 = 11.138 - 1 * np.log10(h)
    gamma1 = 7.096
    gamma2 = 0.201
    label = "Dvornik+23"
    data = "KiDS + GAMA"
    method = "SMF + GGL + clustering"
    min_log10_M = 11.3

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
    label = "Dvornik+20"
    data = "KiDS + GAMA"
    min_log10_M = 10.5

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
    label = "Hudson+15"
    data = "CFHTLenS"
    method = "GGL"
    min_log10_M = 11.3

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
    label = "Moster+10"
    data = "SDSS DR2/DR3 + GADGET-2"
    method = "AM"  # Abundance matching
    min_log10_M = 10.5

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        x = (10 ** np.asarray(log10_M, dtype=float)) / (10 ** cls.log10_M1)
        return np.log10(
            2.0 * cls.ratio_0 / (x ** (-cls.beta) + x ** cls.gamma)
        )


class Moster2018(SHMR):
    fb = 0.156
    log10_M1 = 11.78
    epsN = 0.15
    beta = 1.78
    gamma = 0.57
    label = "Moster+18"
    data = "EMERGE"
    method = "EM" # Emperical model
    min_log10_M = 10.5

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        x = (10 ** np.asarray(log10_M, dtype=float)) / (10 ** cls.log10_M1)
        eps = 2.0 * cls.epsN / (x ** (-cls.beta) + x ** cls.gamma)
        return np.log10(cls.fb * eps)


class Zu2015(SHMR):
    label = "Zu+15"
    data = "SDSS DR7"
    method = "GGL + clustering"
    min_log10_M = 11.0

    @classmethod
    def _lgMh_of_lgMs(cls, lgM_star):
        # Eq. (51): <lg M_h | M_*> vs lg M_* (M_h in h^-1 Msun, M_* in h^-2 Msun)
        return (
            4.41 / (1.0 + np.exp(-1.82 * (lgM_star - 11.18)))
            + 11.12 * np.sin(-0.12 * (lgM_star - 23.37))
        )

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        log10_M = np.asarray(log10_M, dtype=float)
        lg_h = np.log10(h)

        _lm_star = np.linspace(4.0, 12.5, 800)
        _lm_h = cls._lgMh_of_lgMs(_lm_star) - lg_h

        interp = interp1d(
            _lm_h, _lm_star,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        lgM_star = interp(log10_M)

        # Below lg M_h = 11, extend with the slope at lg M_h = 11.
        x0 = 11.0
        y0 = interp(x0)
        eps = 1e-3
        slope = (interp(x0 + eps) - interp(x0 - eps)) / (2.0 * eps)
        lgM_star = np.where(
            log10_M < x0,
            y0 + slope * (log10_M - x0),
            lgM_star,
        )

        return lgM_star - 2.0 * lg_h - log10_M


class Behroozi2019(SHMR):
    EFF_0, EFF_0_A, EFF_0_A2, EFF_0_Z = -1.434595, 1.831346, 1.368294, -0.216943
    M_1, M_1_A, M_1_A2, M_1_Z = 12.03538, 4.556205, 4.417054, -0.731372
    ALPHA, ALPHA_A, ALPHA_A2, ALPHA_Z = 1.963342, -2.315609, -1.732084, 0.177598
    BETA, BETA_A, BETA_Z = 0.481788, -0.840580, -0.470653
    DELTA = 0.410851
    GAMMA, GAMMA_A, GAMMA_Z = -1.034197, -3.100399, -1.054511
    label = "Behroozi+19"
    data = "UniverseMachine"
    method = "EM"
    min_log10_M = 10.5

    @classmethod
    def shmr(cls, log10_M: np.ndarray, z: float=0.1) -> np.ndarray:
        a = 1.0 / (1.0 + z)
        a1 = a - 1.0
        lna = np.log(a)

        logM1 = (
            cls.M_1 + a1 * cls.M_1_A - lna * cls.M_1_A2 + z * cls.M_1_Z
        )
        alpha = (
            cls.ALPHA + a1 * cls.ALPHA_A - lna * cls.ALPHA_A2 + z * cls.ALPHA_Z
        )
        beta = cls.BETA + a1 * cls.BETA_A + z * cls.BETA_Z
        delta = cls.DELTA
        gamma = 10.0 ** (cls.GAMMA + a1 * cls.GAMMA_A + z * cls.GAMMA_Z)

        x = np.asarray(log10_M, dtype=float) - logM1
        logMstar = (
            logM1 + cls.EFF_0 + a1 * cls.EFF_0_A
            - lna * cls.EFF_0_A2 + z * cls.EFF_0_Z
            - np.log10(10.0 ** (-alpha * x) + 10.0 ** (-beta * x))
            + gamma * np.exp(-0.5 * (x / delta) ** 2)
        )
        return logMstar - np.asarray(log10_M, dtype=float)


class Yang2012(SHMR):
    log10_M0 = 10.36 - 2.0 * np.log10(h)
    log10_M1 = 11.06 - 1.0 * np.log10(h)
    alpha = 0.27
    beta = 4.34
    label = "Yang+12"
    data = "SDSS DR7"
    method = "SMF + CSMF"
    min_log10_M = 11.0

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        M = 10 ** np.asarray(log10_M, dtype=float)
        M1 = 10 ** cls.log10_M1
        log10_M_star = (
            cls.log10_M0
            + (cls.alpha + cls.beta) * np.log10(M / M1)
            - cls.beta * np.log10(1.0 + M / M1)
        )
        return log10_M_star - np.asarray(log10_M, dtype=float)


class Shao2026(SHMR):
    log10_Mp = 12.03
    eps = -1.59
    alpha = 2.08
    beta = 0.32
    log10_gamma = -2.80
    delta = 1.03
    label = "Shao+26"
    data = "DESI DR1"
    method = "SMF + GGL + clustering"
    min_log10_M = 11.0

    @classmethod
    def shmr(cls, log10_M: np.ndarray) -> np.ndarray:
        # FIXME: It should be log10_M + np.log10(h) according to the equation.
        # x = np.asarray(log10_M + np.log10(h), dtype=float) - cls.log10_Mp
        # FIXME: However, it does not reproduce the curve in the paper.
        x = np.asarray(log10_M, dtype=float) - cls.log10_Mp
        gamma = 10.0 ** cls.log10_gamma
        log10_M_star = (
            cls.eps
            - np.log10(10.0 ** (-cls.alpha * x) + 10.0 ** (-cls.beta * x))
            + gamma * np.exp(-0.5 * (x / cls.delta) ** 2)
        ) + cls.log10_Mp - 2.0 * np.log10(h)
        return log10_M_star - log10_M


