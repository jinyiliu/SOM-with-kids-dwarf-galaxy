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
        rp: float | np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    """Analytical NFW excess surface density profile.

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun
        z_lens: Lens redshift.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
    """
    if (c is None) == (f_c is None):
        raise ValueError("Provide exactly one of: c or f_c.")

    if c is not None:
        conc = ConcentrationConstant(c=c, mass_def=MassDef200m)
    else:
        conc = ConcentrationDuffy08(fc_bar=f_c, mass_def=MassDef200m)

    a = 1. / (1. + z_lens)
    nfw = HaloProfileNFW(
        mass_def=MassDef200m,
        concentration=conc,
        truncated=False,
        projected_analytic=True,
        cumul2d_analytic=True,
    )

    rp_Mpc = rp / h  # Mpc/h -> physical Mpc
    Sigma = nfw.projected(Planck18, rp_Mpc, 10**log10_M, a)  # M_sun / Mpc2
    Sigma_bar = nfw.cumul2d(Planck18, rp_Mpc, 10**log10_M, a)  # M_sun / Mpc2

    return (Sigma_bar - Sigma) / 1.e12

def DSigma_1h_sm():
    pass

def DSigma_2h():
    pass