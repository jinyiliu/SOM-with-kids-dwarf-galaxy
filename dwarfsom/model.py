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
        log10_M: log10 host halo mass M_200m [M_sun].
        z_lens: Lens redshift.
        c: Concentration. Mutually exclusive with f_c.
        f_c: Duffy08 amplitude. Mutually exclusive with c.
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