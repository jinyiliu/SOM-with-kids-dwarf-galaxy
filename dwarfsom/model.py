import numpy as np
import pyccl as ccl
from pyccl.halos import (
    HaloProfileNFW,
    ConcentrationConstant,
    ConcentrationDuffy08,
    MassDef200m,
)
from astropy.cosmology import Planck18 as astropy_Planck18

h = astropy_Planck18.H0.value / 100

Planck18 = ccl.Cosmology(
    Omega_c=(astropy_Planck18.Om0 - astropy_Planck18.Ob0),
    Omega_b=astropy_Planck18.Ob0,
    h=h,
    T_CMB=astropy_Planck18.Tcmb0.value,
    Neff=astropy_Planck18.Neff,
    n_s=0.9649,
    sigma8=0.8111,
)


def DSigma():
    pass

def DSigma_1h_cm(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    return DSigma_NFW(
        rp, log10_M, z_lens, c, f_c, truncated=False, analytic=True)


def DSigma_1h_sm_sub(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    return DSigma_NFW(
        rp, log10_M, z_lens, c, f_c, truncated=True, analytic=False)


def DSigma_1h_sm_host():
    pass


def DSigma_2h():
    pass


def DSigma_NFW(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
        truncated: bool=False,
        analytic: bool=False,
):
    """Analytical NFW excess surface density profile.

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
        truncated: Truncation.
        analytic: Whether to use analytical algorithm.
    """
    nfw, a, M = _get_nfw_profile(
        log10_M, z_lens, c, f_c, truncated, analytic)

    rp_Mpc = rp / h  # Mpc/h -> physical Mpc
    Sigma = nfw.projected(Planck18, rp_Mpc, M, a)
    Sigma_bar = nfw.cumul2d(Planck18, rp_Mpc, M, a)

    return (Sigma_bar - Sigma) / 1.e12


def DSigma_NFW_offset(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        rp_sat: float,
        c: float | None=None,
        f_c: float | None=None,
):
    """Off-centre NFW excess surface density.

    DeltaSigma at projected separation `rp` from a satellite
    offset by `rp_sat` from the host NFW centre.

    Uses azimuthal averaging over phi, then cumulative radial
    integration for the mean enclosed surface density.

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of host halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        rp_sat: Projected offset of satellite from host centre in Mpc/h.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
    """
    if rp_sat == 0:
        return DSigma_NFW(
            rp, log10_M, z_lens, c, f_c, truncated=False, analytic=True)

    nfw, a, M = _get_nfw_profile(
        log10_M, z_lens, c, f_c, truncated=False, analytic=True)

    rp = np.atleast_1d(rp) / h
    rp_sat = rp_sat / h

    _n_fine = max(300, 4 * len(rp))
    _r_min = max(np.min(rp) * 0.3, 1e-5)
    _r_max = max(np.max(rp) * 1.5, rp_sat + np.max(rp))
    r_fine = np.logspace(np.log10(_r_min), np.log10(_r_max), _n_fine)

    phi = np.linspace(0, 2 * np.pi, 80)

    Sigma_fine = np.array([
        np.mean(nfw.projected(
            Planck18,
            np.sqrt(rp_sat**2 + r**2 + 2 * rp_sat * r * np.cos(phi)),
            M,
            a,
        ))
        for r in r_fine
    ])

    rSigma = r_fine * Sigma_fine
    I_cum = np.zeros_like(rSigma)
    dr = r_fine[1:] - r_fine[:-1]
    I_cum[1:] = 0.5 * np.cumsum(dr * (rSigma[1:] + rSigma[:-1]))
    Sigma_bar = 2.0 * I_cum / r_fine**2
    Sigma_bar[0] = Sigma_fine[0]

    dSigma_fine = (Sigma_bar - Sigma_fine) / 1e12
    return np.interp(rp, r_fine, dSigma_fine)


def _get_nfw_profile(
        log10_M: float,
        z_lens: float,
        c: float,
        f_c: float,
        truncated: bool,
        analytic: bool,
) -> tuple[HaloProfileNFW, float, float]:
    """Build HaloProfileNFW instance.

    Args:
        log10_M: log10 of host halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
        truncated:
        analytic:
    """
    if (c is None) == (f_c is None):
        raise ValueError("Provide exactly one of: c or f_c.")

    if c is not None:
        conc = ConcentrationConstant(c=c, mass_def=MassDef200m)
    else:
        conc = ConcentrationDuffy08(fc_bar=f_c, mass_def=MassDef200m)

    a = 1.0 / (1.0 + z_lens)
    nfw = HaloProfileNFW(
        mass_def=MassDef200m,
        concentration=conc,
        truncated=truncated,
        projected_analytic=analytic,
        cumul2d_analytic=analytic,
    )
    return nfw, a, 10 ** log10_M


def _satellite_HOD(
        log10_M: float | np.ndarray,
        log10_M0: float=11.0,
        log10_M1: float=12.0,
        alpha: float=1.0,
):
    """Power-law satellite occupation number with a hard cutoff.

    log10⟨N_sat⟩ = alpha x (log10(M - M0) - log10(M1))
    if M > M0, else 0

    Args:
        log10_M: log10 of host halo mass M_200m in M_sun.
        log10_M0: log10 minimum host mass for satellites.
        log10_M1: log10 mass where ⟨N_sat⟩ = 1.
        alpha: Power-law slope.
    """
    _log10_M = np.atleast_1d(np.asarray(log10_M, dtype=float))
    N = np.zeros_like(_log10_M)
    mask = _log10_M > log10_M0
    N[mask] = 10. ** (
        alpha * (
            np.log10(10.**_log10_M[mask] - 10.**log10_M0)
            - log10_M1
        )
    )
    if np.ndim(log10_M) == 0:
        return float(N[0])
    return N