import os
import pickle as pk
import warnings

import numpy as np
import pyccl as ccl
from pyccl.halos import (
    HaloProfileNFW,
    ConcentrationConstant,
    ConcentrationDuffy08,
    MassDef200m,
)
from astropy.cosmology import Planck18 as astropy_Planck18
import astropy.units as u
from astropy.constants import G
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

h = astropy_Planck18.H0.value / 100

_cache_dir = "/net/alblas/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL"

if os.path.exists(
    os.path.join(_cache_dir, "_cache_host_terms.pk")
):
    with open(os.path.join(_cache_dir, "_cache_host_terms.pk"), "rb") as f:
        _cache_host_terms = pk.load(f)
else:
    _cache_host_terms = {}


_Omega_m = astropy_Planck18.Om0
_H0_100 = (100 * u.km / u.s / u.Mpc).to(u.s**-1)
_rho_crit0 = (
    3. * _H0_100**2 / (
        8 * np.pi * G.to(u.Mpc**3 / (u.M_sun * u.s**2))
    )
).value * h**2  # M_sun / (comoving Mpc)^3


Planck18 = ccl.Cosmology(
    Omega_c=(astropy_Planck18.Om0 - astropy_Planck18.Ob0),
    Omega_b=astropy_Planck18.Ob0,
    h=h,
    T_CMB=astropy_Planck18.Tcmb0.value,
    Neff=astropy_Planck18.Neff,
    n_s=0.9649,
    sigma8=0.8111,
)

# Quintic smoothstep function
smoothstep = lambda y: y ** 3 * (10.0 - 15.0 * y + 6.0 * y ** 2)


def DSigmaModel(
        rp: np.ndarray,
        z_lens: float,
        log10_M: float,
        f_c: float,
        frac_sat: float,
        log10_M_star: float,
        R: float=1.,
        tau: float | None=None,
        log10_M_host: float | None=None,
        use_single_halo: bool=False,
        return_components: bool=False,
):
    ds_1h_cm = DSigma_1h_cm(rp, log10_M, z_lens, f_c=f_c)
    ds_1h_sm_sub = DSigma_1h_sm_sub(rp, log10_M, z_lens, f_c=1., tau=tau)
    ds_1h_sm_host = DSigma_1h_sm_host(
        rp, z_lens,
        f_c=1.,
        R=R,
        log10_M_star=log10_M_star,
        log10_M_host=log10_M_host,
        use_single_halo=use_single_halo,
    )

    if use_single_halo:
        raise ValueError(
            "log10_M_host must be provided when use_single_halo=True"
        )
    else:
        key = (z_lens, None, f_c, log10_M_star)
        if key not in _cache_host_terms:
            _cache_host_terms[key] = (
                precompute_host_dsigma_terms(
                    z_lens, None, f_c, log10_M_star=log10_M_star,
                    n_rs=70, use_single_halo=False
                )
            )
        log10_M_host_mean = _cache_host_terms[key]["log10_M_host_mean"]

    ds_2h = DSigma_2h(rp, log10_M, z_lens, log10_M_host_mean, frac_sat)
    ds_star = DSigma_star(rp, log10_M_star)
    ds_total = (
            ds_star +
            (1 - frac_sat) * ds_1h_cm +
            frac_sat * (ds_1h_sm_sub + ds_1h_sm_host) +
            ds_2h
    )
    if return_components:
        return ds_total, {
            "star": ds_star,
            "1h_cm": (1 - frac_sat) * ds_1h_cm,
            "1h_sm_sub": frac_sat * ds_1h_sm_sub,
            "1h_sm_host": frac_sat * ds_1h_sm_host,
            "2h": ds_2h,
            "total": ds_total,
        }
    return ds_total


def DSigma_1h_cm(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    return DSigma_NFW(
        rp, log10_M, z_lens, c, f_c, truncated=False, analytic=True)


def DSigma_star(
        rp: np.ndarray,
        log10_M_star: float,
):
    """Stellar point mass excess surface density.

    ΔΣ_star(R) = M_star / (π R²)
    """
    M_star = 10.0 ** log10_M_star
    rp_pc = np.atleast_1d(np.asarray(rp, dtype=float)) * 1.e6 / h  # comoving pc
    return M_star / (np.pi * rp_pc ** 2) / h   # h M_sun / (comoving pc)^2


def DSigma_1h_sm_sub(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
        tau: float | None=None,
):
    """BMO smoothly-truncated NFW subhalo ESD (Baltz et al. 2009).

    Args:
        rp: Projected separations in comoving Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        c: Concentration. Mutually exclusive with f_c.
        f_c: Duffy08 amplitude. Mutually exclusive with c.
        tau: r_t / r_s (dimensionless truncation parameter). Default: 2.5.
    """
    if (c is None) == (f_c is None):
        raise ValueError("Provide exactly one of: c or f_c.")

    a = 1.0 / (1.0 + z_lens)
    M = 10.0 ** log10_M
    r_vir = MassDef200m.get_radius(Planck18, M, a)  # physical Mpc
    r_vir = r_vir / a   # comoving Mpc

    if c is not None:
        conc_val = c
    else:
        conc = ConcentrationDuffy08(fc_bar=f_c, mass_def=MassDef200m)
        conc_val = float(conc(Planck18, M, a))

    r_s = r_vir / conc_val  # comoving Mpc

    if tau is None:
        tau_val = 2.5  # default: r_t = 2.5 * r_s
    else:
        tau_val = tau

    # M0 = M / f(c) to convert M_200m to M0
    M0 = M / (np.log(1 + conc_val) - conc_val / (1 + conc_val))

    rp_Mpc = np.atleast_1d(np.asarray(rp, dtype=float)) / h
    # comoving Mpc/h -> comoving Mpc

    r_min = max(np.min(rp_Mpc) * 0.3, 1e-5)
    r_max = np.max(rp_Mpc) * 1.5
    r_fine = np.logspace(np.log10(r_min), np.log10(r_max), 200)

    Sigma_fine = _Sigma_BMO(r_fine, M0, r_s, tau_val)   # M_sun / (comoving Mpc)^2
    rS = r_fine * Sigma_fine
    I_cum = np.zeros_like(rS)
    dr = np.diff(r_fine)
    I_cum[1:] = 0.5 * np.cumsum(dr * (rS[1:] + rS[:-1]))

    # Compute Σ̄(<r_fine[0])
    r_sub = np.logspace(
        start=np.log10(max(r_s * 1e-3, 1e-5)),
        stop=np.log10(r_fine[0]),
        num=100,
    )
    Sigma_sub = _Sigma_BMO(r_sub, M0, r_s, tau_val)
    I_0 = np.trapezoid(r_sub * Sigma_sub, r_sub)

    I_cum += I_0
    Sigma_bar = 2.0 * I_cum / r_fine**2
    Sigma_bar[0] = 2.0 * I_0 / r_fine[0]**2

    ds_fine = (Sigma_bar - Sigma_fine) / 1.e12 / h
    return np.interp(rp_Mpc, r_fine, np.maximum(ds_fine, 0.0))


def DSigma_1h_sm_host(
        rp: np.ndarray,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
        R: float=1.,
        log10_M_star: float | None=None,
        log10_M_host: float | None=None,
        use_single_halo: bool=False,
):
    """Host halo contribution to satellite-matter ESD.

    Integrates over the halo mass function, satellite HOD, and
    satellite radial distribution to compute the average host
    dark matter ESD around satellite galaxies.

    No free parameters — uses fixed HOD and Duffy08 c(M).

    Args:
        rp: Projected separations in Mpc/h.
        z_lens: Lens redshift.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
        R: Concentration ratio between the satellite distribution and
        log10_M_star: Satellite stellar mass.
        log10_M_host: Host halo mass.
        use_single_halo:
    """
    rp_grid, ds_grid = compute_host_dsigma(
        z_lens, c, f_c, R, log10_M_star, log10_M_host, use_single_halo)
    return interp1d(
        rp_grid, ds_grid, kind="cubic",
        bounds_error=False, fill_value=(ds_grid[0], ds_grid[-1]),
    )(rp)


def DSigma_2h(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
        log10_M_host: float | None=None,
        frac_sat: float | None=None,
):
    """Two-halo ESD from linear bias with Tinker 2005 scale dependence.

    ξ_gm(r) = b(M) × η(r) × ξ_lin(r)

    The projected surface density Σ(R) is obtained via Abel
    integral, then ΔΣ_2h = Σ̄(<R) − Σ(R).

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        log10_M_host:
        frac_sat:
    """
    rp_grid, ds_grid = compute_2h(z_lens, log10_M, log10_M_host, frac_sat)
    return np.interp(np.atleast_1d(rp), rp_grid, ds_grid)


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
        rp: Projected separations in comoving Mpc/h.
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

    rp_Mpc = rp / h # comoving Mpc/h -> transverse comoving Mpc
    Sigma = nfw.projected(Planck18, rp_Mpc, M, a)   # M_sun / (comoving Mpc)^2
    Sigma_bar = nfw.cumul2d(Planck18, rp_Mpc, M, a)
    ret = (Sigma_bar - Sigma) / 1.e12 / h   # h M_sun / (comoving pc)^2

    return ret


def _Sigma_NFW_offset(
        rp: np.ndarray,
        rp_sat: float,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    """Off-centre NFW surface density.

    Sigma at projected separation `rp` from a satellite
    offset by `rp_sat` from the host NFW centre.

    Uses azimuthal averaging over phi, then cumulative radial
    integration for the mean enclosed surface density.

    Args:
        rp: Projected separations in comoving Mpc/h.
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

    rp = np.atleast_1d(rp) / h  # comoving Mpc/h -> transverse comoving Mpc
    rp_sat = rp_sat / h

    _n_fine = max(300, 4 * len(rp))
    _r_min = max(np.min(rp) * 0.3, 1e-5)
    _r_max = max(np.max(rp) * 1.5, rp_sat + np.max(rp))
    r_fine = np.logspace(np.log10(_r_min), np.log10(_r_max), _n_fine)

    n_phi = 160
    phi = np.linspace(0, 2 * np.pi, n_phi)

    Sigma_fine = np.array([
        np.mean(nfw.projected(
            Planck18,
            np.sqrt(rp_sat**2 + r**2 + 2 * rp_sat * r * np.cos(phi)),
            M,
            a,
        ))
        for r in r_fine
    ])  # M_sun / (comoving Mpc)^2
    Sigma_fine = Sigma_fine / 1.e12 / h # h M_sun / (comoving pc)^2
    return interp1d(
        r_fine, Sigma_fine, kind="cubic",
        bounds_error=False, fill_value=(Sigma_fine[0], Sigma_fine[-1]),
    )(rp)


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


def _F_bmo(x: np.ndarray):
    """Equation A.5 in Baltz et al. (2009)

        F(x) = cos^-1(1/x) / sqrt(x^2 - 1)

    where x is defined as r/r_s. r_s is the scale radius.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.empty_like(x)

    lo = x < 1.0
    result[lo] = np.arccosh(1.0 / x[lo]) / np.sqrt(1.0 - x[lo]**2)

    hi = x > 1.0
    result[hi] = np.arccos(1.0 / x[hi]) / np.sqrt(x[hi]**2 - 1.0)

    result[x == 1.0] = 1.0
    return result


def _L_bmo(x: np.ndarray, tau: float):
    """Equation A.6 in Baltz et al. (2009)

        L(x, tau) = ln(x / (sqrt(tau^2 + x^2) + tau))

    where x is defined as r/r_s. r_s is the scale radius.
    tau is the ratio of the truncation radius to scale radius.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.log(x / (np.sqrt(tau**2 + x**2) + tau))


def _Sigma_BMO(R: np.ndarray, M0: float, rs: float, tau: float):
    """Equation A.7 in Baltz et al. (2009)

    Args:
        R: Projected radii in comoving Mpc.
        M0: Profile normalisation mass in M_sun.
        r_s: Scale radius in comoving Mpc.
        tau: tau = r_t / r_s (truncation parameter).
    """
    x = np.atleast_1d(np.asarray(R / rs, dtype=float))

    tau2 = tau ** 2
    tau2p1 = tau2 + 1.0
    x2 = x**2
    sqrt_tau2_plus_x2 = np.sqrt(tau2 + x2)

    Fx = _F_bmo(x)
    Lx = _L_bmo(x, tau)

    pref = M0 / rs ** 2 * tau2 / (2.0 * np.pi * tau2p1 ** 2)

    term1 = tau2p1 / (x2 - 1.0) * (1.0 - Fx)
    term1[np.abs(x - 1.0) < 1e-10] = tau2p1 / 3.0

    term2 = 2.0 * Fx
    term3 = -np.pi / sqrt_tau2_plus_x2
    term4 = (tau2 - 1.0) / (tau * sqrt_tau2_plus_x2) * Lx

    return pref * (term1 + term2 + term3 + term4)   # M_sun / (comoving Mpc)^2


def _Sigma_BMO_offset(
        rp: np.ndarray,
        rp_sat: float,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    """Off-centre BMO (truncated NFW) surface density.

    Args:
        rp: Projected separations in comoving Mpc/h.
        log10_M: log10 of host halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        rp_sat: Projected offset of satellite from host centre in Mpc/h.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Duffy08 amplitude. Mutually exclusive with c.

    Returns:
        Surface density in h M_sun / (comoving pc)^2.
    """
    if (c is None) == (f_c is None):
        raise ValueError("Provide exactly one of: c or f_c.")

    a = 1.0 / (1.0 + z_lens)
    M = 10.0 ** log10_M
    r_vir = MassDef200m.get_radius(Planck18, M, a) / a  # comoving Mpc

    if c is not None:
        conc_val = c
    else:
        conc = ConcentrationDuffy08(fc_bar=f_c, mass_def=MassDef200m)
        conc_val = float(conc(Planck18, M, a))

    r_s = r_vir / conc_val                              # comoving Mpc
    tau_val = conc_val

    # M0 = M / f(c) to convert M_200m to M0 (same as DSigma_1h_sm_sub)
    M0 = M / (np.log(1 + conc_val) - conc_val / (1 + conc_val))

    rp = np.atleast_1d(np.asarray(rp, dtype=float)) / h   # comoving Mpc
    rp_sat = rp_sat / h

    if rp_sat == 0:
        return _Sigma_BMO(rp, M0, r_s, tau_val) / 1.e12 / h

    _n_fine = max(300, 4 * len(rp))
    _r_min = max(np.min(rp) * 0.3, 1e-5)
    _r_max = max(np.max(rp) * 1.5, rp_sat + np.max(rp))
    r_fine = np.logspace(np.log10(_r_min), np.log10(_r_max), _n_fine)

    n_phi = 160
    phi = np.linspace(0, 2 * np.pi, n_phi)

    Sigma_fine = np.array([
        np.mean(_Sigma_BMO(
            np.sqrt(rp_sat**2 + r**2 + 2 * rp_sat * r * np.cos(phi)),
            M0, r_s, tau_val,
        ))
        for r in r_fine
    ])                                                  # M_sun / (comoving Mpc)^2
    Sigma_fine = Sigma_fine / 1.e12 / h                 # h M_sun / (comoving pc)^2
    return interp1d(
        r_fine, Sigma_fine, kind="cubic",
        bounds_error=False, fill_value=(Sigma_fine[0], Sigma_fine[-1]),
    )(rp)


def _Sigma_NFW_hollow(
        R: np.ndarray,
        log10_M: float,
        z_lens: float,
        r_hollow: float,
        r_out: float | None=None,
        c: float | None=None,
        f_c: float | None=None,
        n_l: int=256,
):
    """Projected surface density of a smoothly-hollowed, virial-truncated NFW.

    The 3D satellite tracer is
        n(r) = ρ_NFW(r) × smoothstep(r / r_hollow)   for  r < r_vir,
        n(r) = 0                                     otherwise,
    where smoothstep rises from 0 at r = 0 to 1 at r = r_hollow.
    Projecting gives
        Σ(R) = 2 ∫ ρ_NFW(√(R² + l²)) smoothstep(√(R²+l²)/r_hollow) dl.

    Args:
        R: Projected radii in comoving Mpc.
        log10_M: log10 of host halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        r_hollow: Radius of the hollow core in comoving Mpc.
        r_out: Truncating radius in comoving Mpc.
        c: Concentration (r_200m / r_s). Mutually exclusive with f_c.
        f_c: Amplitude scaling of the Duffy2008 c(M,z) relation. Mutually
            exclusive with c.
        n_l: Number of line-of-sight quadrature points.

    Returns:
        Projected surface density in M_sun / (comoving Mpc)^2.
    """
    nfw, a, M = _get_nfw_profile(
        log10_M, z_lens, c, f_c, truncated=False, analytic=True)

    if r_out is None: # truncate at virial radius
        r_vir = MassDef200m.get_radius(Planck18, M, a) / a  # comoving Mpc
        r_out = r_vir

    R = np.atleast_1d(np.asarray(R, dtype=float))

    Sigma = np.zeros_like(R)
    for i, Ri in enumerate(R):
        if Ri >= r_out:
            continue
        l_max = np.sqrt(r_out**2 - Ri**2)               # outer truncation
        l = np.linspace(0.0, l_max, n_l)
        r = np.sqrt(Ri**2 + l**2)
        rho = nfw.real(Planck18, r, M, a)               # 3D NFW density
        hollow = smoothstep(np.clip(r / r_hollow, 0.0, 1.0))
        Sigma[i] = 2.0 * np.trapezoid(rho * hollow, l)

    return Sigma


def _satellite_radial_distribution(
        rp_sat: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
        density: bool=False,
):
    """Projected satellite number density profile.

    Satellites trace the host NFW projected mass:
    P(r_p_sat | M) is propotional to 2π rp_sat x Σ_NFW(rp_sat | M)

    Args:
        rp_sat: Satellite distance to host halo centre in comoving Mpc.
        log10_M: log10 of host halo mass in M_sun.
        z_lens: Lens redshift.
        c:
        f_c:
        density:
    """
    r_Mpc = np.atleast_1d(np.asarray(rp_sat, dtype=float))  # comoving Mpc

    a = 1. / (1. + z_lens)
    r_vir = MassDef200m.get_radius(Planck18, 10**log10_M, a) / a # comoving Mpc
    Sigma = _Sigma_NFW_hollow( # M_sun / (comoving Mpc)^2
        r_Mpc, log10_M, z_lens,
        r_hollow=0.01,
        r_out=1. * r_vir,
        c=c,
        f_c=f_c,
    )

    P = 2 * np.pi * r_Mpc * Sigma

    if density:
        normalization = np.trapezoid(P, r_Mpc)
        if normalization <= 0.0 or not np.isfinite(normalization):
            raise ValueError("Satellite radial distribution is invalid.")
        P /= normalization
    else:
        widths = np.zeros_like(r_Mpc)
        widths[1:-1] = (r_Mpc[2:] - r_Mpc[:-2]) / 2
        widths[0] = (r_Mpc[1] - r_Mpc[0]) / 2
        widths[-1] = (r_Mpc[-1] - r_Mpc[-2]) / 2
        P *= widths

        # Force a smooth truncation if needed
        # x = (r_Mpc - r_Mpc[0]) / (r_Mpc[-1] - r_Mpc[0])
        # x_max = x[np.argmax(P)]
        # window = np.ones_like(x)
        # if x_max < 1.:
        #     right = x > x_max
        #     window[right] = smoothstep((1. - x[right]) / (1. - x_max))
        #
        # P *= window

        P /= P.sum()

    return P


def compute_host_dsigma(
        z_lens, c, f_c,
        R: float=1.,
        log10_M_star: float | None=None,
        log10_M_host: float | None=None,
        use_single_halo: bool=False,
):
    """Precompute the host halo ESD contribution for given z_lens."""
    # FIXME: Include n_rs in the function parameters
    n_rs = 70

    if use_single_halo:
        if log10_M_host is None:
            raise ValueError(
                "log10_M_host must be provided when use_single_halo=True.")

        if R is not None:
            warnings.warn(
                "R is only supported when use_single_halo=False for now. "
                "Ignoring R in the following calculations.",
                UserWarning
            )

        key = (z_lens, c, f_c)
        if key not in _cache_host_terms:
            _cache_host_terms[key] = precompute_host_dsigma_terms(
                z_lens, c, f_c,
                n_rs=n_rs,
                n_log10_M_host=100,
                use_single_halo=True,
            )

        cache = _cache_host_terms[key]
        lm_grid = cache["log10_M_host_grid"]
        Sigma_M = cache["Sigma_M"] # M_sun / (comoving pc)^2
        rp_out_Mpc = cache["rp_out_Mpc"] # comoving Mpc

        idx = np.searchsorted(lm_grid, log10_M_host) - 1
        frac = (log10_M_host - lm_grid[idx]) / (lm_grid[idx + 1] - lm_grid[idx])

        result_Sigma = (1 - frac) * Sigma_M[idx] + frac * Sigma_M[idx + 1]

    else:
        if log10_M_star is None:
            raise ValueError(
                "log10_M_star must be provided when use_single_halo=False.")

        key = (z_lens, c, f_c, log10_M_star)
        if key not in _cache_host_terms:
            _cache_host_terms[key] = (
                precompute_host_dsigma_terms(
                    z_lens, c, f_c, log10_M_star=log10_M_star,
                    n_rs=n_rs, use_single_halo=False
                )
            )

        R_grid = _cache_host_terms[key]["R_grid"]
        Sigma_R = _cache_host_terms[key]["Sigma_R"]
        result_Sigma = 10 ** interp1d(
            np.log10(R_grid), np.log10(Sigma_R),
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(np.log10(R))

        rp_out_Mpc = np.logspace(-5, 2, 100)  # comoving Mpc


    # --- Σ̄ → ΔΣ (shared pipeline) ---
    rp_out_Mpc_com = rp_out_Mpc * h  # Mpc/h
    R_Sigma = rp_out_Mpc_com * result_Sigma
    I_cum = np.zeros_like(R_Sigma)
    dr = np.diff(rp_out_Mpc_com)
    I_cum[1:] = 0.5 * np.cumsum(dr * (R_Sigma[1:] + R_Sigma[:-1]))
    Sigma_bar = 2.0 * I_cum / rp_out_Mpc_com ** 2
    Sigma_bar[0] = result_Sigma[0]

    ds_pop = np.maximum((Sigma_bar - result_Sigma), 0.0)
    return rp_out_Mpc_com, ds_pop


def precompute_host_dsigma_terms(
        z_lens, c, f_c,
        log10_M_star: float | None=None,
        n_rs: int=70,
        log10_M_host_min: float=12.,
        log10_M_host_max: float=16.,
        n_log10_M_host: int=50,
        use_single_halo: bool=False,
):
    a = 1.0 / (1.0 + z_lens)
    rp_out_Mpc = np.logspace(-5, 2, 100)  # comoving Mpc

    if use_single_halo:
        # Compute the surface density of a grid of host halo masses
        # averaged over the satellite radial distribution
        log10_M_host_grid = np.linspace(
            log10_M_host_min, log10_M_host_max, n_log10_M_host)

        Sigma_M = np.zeros(
            (n_log10_M_host, len(rp_out_Mpc)))

        Sigma_off_rs = np.zeros((n_rs, len(rp_out_Mpc)))

        for i, lm in enumerate(log10_M_host_grid):
            r_vir = MassDef200m.get_radius(
                Planck18, 10**lm, a
            ) / a   # comoving Mpc
            rs_grid = np.logspace(
                np.log10(0.01 * r_vir),
                np.log10(1. * r_vir),
                num=n_rs,
            )

            for j, rs in enumerate(rs_grid):
                Sigma_off_rs[j] = _Sigma_NFW_offset(
                    rp=rp_out_Mpc * h,
                    rp_sat=rs * h,
                    log10_M=lm,
                    z_lens=z_lens,
                    c=c,
                    f_c=f_c,
                )   # h M_sun / (comoving pc)^2

            P_rs = _satellite_radial_distribution(
                rs_grid, lm, z_lens, c=c, f_c=f_c)
            _Sigma_M = np.sum(P_rs[:, None] * Sigma_off_rs, axis=0)
            Sigma_M[i] = _Sigma_M

            # FIXME: Adjust the smoothing window size
            # Svitzky-Golay smoothing of log10 Sigma_M: suppress interpolation
            # noise in the host term so the projected dSigma stays smooth
            # log_Sigma = np.log10(_Sigma_M)
            # Sigma_M[i] = 10.0 ** savgol_filter(
            #     log_Sigma,
            #     window_length=31,
            #     polyorder=3,
            # )

        return {
            "log10_M_host_grid": log10_M_host_grid,
            "Sigma_M": Sigma_M,
            "rp_out_Mpc": rp_out_Mpc,
        }

    else: # Average over HMF and HOD
        from pyccl.halos import MassFuncTinker10

        # Concentration ratio grid
        R_grid = np.linspace(0.01, 1.2, 300)

        # Host halo mass grid
        _log10_M_host = np.linspace(start=11., stop=17., num=100)
        log10_M_host = 0.5 * (_log10_M_host[1:] + _log10_M_host[:-1])
        dlog10_M_host = _log10_M_host[1:] - _log10_M_host[:-1]

        # Probability of host halo mass provided the stellar mass
        P_log10_M_host = _host_halo_mass_pdf(z_lens, log10_M_star, log10_M_host, dlog10_M_host)

        # Surface density of parameters (M_host, rs)
        Sigma_M_host_rs = np.zeros(shape=(
            len(log10_M_host), n_rs, len(rp_out_Mpc)
        ))
        rs_grid_M_host = np.zeros(shape=(len(log10_M_host), n_rs))

        # Loop over host halo mass
        for i, lm in enumerate(log10_M_host):
            r_vir = MassDef200m.get_radius(Planck18, 10**lm, a)  # physical Mpc
            r_vir = r_vir / a  # comoving Mpc

            rs_grid_M_host[i] = np.logspace(
                np.log10(0.01 * r_vir),
                np.log10(1. * r_vir),
                num=n_rs,
            )  # comoving Mpc

            for j, rs in enumerate(rs_grid_M_host[i]):
                Sigma = _Sigma_NFW_offset(
                    rp=rp_out_Mpc * h,  # comoving Mpc/h
                    rp_sat=rs * h,  # comoving Mpc/h
                    log10_M=lm,
                    z_lens=z_lens,
                    c=c,
                    f_c=f_c,
                )  # h M_sun / (comoving pc)^2
                Sigma_M_host_rs[i, j] = Sigma

        Sigma_R = np.zeros(
            (len(R_grid), len(rp_out_Mpc))
        )
        for i, R in enumerate(R_grid):
            result_Sigma = np.zeros(len(rp_out_Mpc))

            for j, lm in enumerate(log10_M_host):
                # Satellite radial distribution given halo mass
                P_rs = _satellite_radial_distribution(
                    rs_grid_M_host[j], lm, z_lens, c=c, f_c=R,
                )
                Sigma_rs = Sigma_M_host_rs[j]  # shape (n_rs, len(rp_out_Mpc))

                result_Sigma += P_log10_M_host[j] * \
                    np.sum(P_rs[:, None] * Sigma_rs, axis=0)

            Sigma_R[i] = result_Sigma

        return {
            "log10_M_host": log10_M_host,
            "log10_M_host_mean": np.average(log10_M_host, weights=P_log10_M_host),
            "dlog10_M_host": dlog10_M_host,
            "P_log10_M_host": P_log10_M_host,
            "rs_grid_M_host": rs_grid_M_host,
            "Sigma_M_host_rs": Sigma_M_host_rs,
            "R_grid": R_grid,
            "Sigma_R": Sigma_R,
        }


def _host_halo_mass_pdf(
        z_lens: float,
        log10_M_star: float,
        log10_M_host: np.ndarray,
        dlog10_M_host: np.ndarray | None=None,
        density: bool=False,
        b1: float | None=None,
        alpha_s: float | None=None,
):
    """Conditonal probability of host halo mass given satellite stellar mass.

    Args:
        z_lens: Lens redshift.
        log10_M_star: log10 of satellite stellar mass in M_sun.
        log10_M_host: log10 of host halo mass grid in M_sun.
        dlog10_M_host: Bin width of the host halo mass grid.
        density: If True, return the normalized probability density.
        b1: Satellite CSMF normalisation slope (Dvornik+23). If None, the
            default in _satellite_CSMF is used.
        alpha_s: Satellite CSMF faint-end slope. If None, the default in
            _satellite_CSMF is used.
    """
    from pyccl.halos import MassFuncTinker10

    a = 1. / (1. + z_lens)
    CSMF_kwargs = {}
    if b1 is not None:
        CSMF_kwargs["b1"] = b1
    if alpha_s is not None:
        CSMF_kwargs["alpha_s"] = alpha_s
    PHI_s = _satellite_CSMF(log10_M_star, log10_M_host, **CSMF_kwargs)
    hmf = MassFuncTinker10(mass_def=MassDef200m)
    dndln_M_host = hmf(Planck18, h * 10 ** log10_M_host, a)

    w = PHI_s * dndln_M_host * np.log(10)

    if density:
        norm = np.trapezoid(w, x=log10_M_host)
        return w / norm
    else:
        if dlog10_M_host is None:
            raise ValueError
        else:
            P = w * dlog10_M_host
            return P / P.sum()


def _satellite_CSMF(
        log10_M_star: float,
        log10_M_host: float | np.ndarray,

        # Dvornik et al. (2023) MAP+PJ-HPD
        # gamma1: float=7.385,
        # gamma2: float=0.201,
        # log10_M0: float=10.521 - 2. * np.log10(h),
        # log10_M1: float=11.145 - 1. * np.log10(h),
        # b0: float=-0.120,
        # b1: float=1.17,
        # alpha_s: float=-0.847,

        # Dvornik et al. (2023) MMAX
        gamma1: float=7.096,
        gamma2: float=0.201,
        log10_M0: float=10.519 - 2. * np.log10(h),
        log10_M1: float=11.138 - 1. * np.log10(h),
        b0: float=-0.024,
        b1: float=1.149,
        alpha_s: float=-0.858,
):
    """Satellite Conditional Stellar Mass Function.

    A modified Schechter function is used for satellite CSMF.


    Returns:
        Average number of galaxies of stellar mass log10_M_star that resides in
        a halo mass of log10_M_host.
    """
    M_star = 10 ** log10_M_star
    log10_M_host_grid = np.atleast_1d(log10_M_host)
    M_host_grid = 10 ** log10_M_host_grid


    M0 = 10 ** log10_M0
    M1 = 10 ** log10_M1
    M_c = M0 * (M_host_grid / M1)**gamma1 / (1 + M_host_grid / M1)**(gamma1 - gamma2)
    M_s = 0.56 * M_c
    log10_phi_s = b0 + b1 * (log10_M_host_grid - 13. + np.log10(h))
    phi_s = 10 ** log10_phi_s

    PHI_s = (M_star / M_s)**alpha_s * np.exp(-(M_star / M_s)**2)
    PHI_s *= phi_s / M_s
    return PHI_s



def compute_2h(
        z_lens,
        log10_M,
        log10_M_host: float | None=None,
        frac_sat: float | None=None,
):
    """Compute the 2-halo ESD for a given (z_lens, log10_M)."""
    from pyccl.halos import HaloBiasTinker10

    a = 1.0 / (1.0 + z_lens)
    M = 10 ** log10_M

    bias_model = HaloBiasTinker10(mass_def=MassDef200m)
    if log10_M_host is not None:
        M_host = 10 ** log10_M_host
        if frac_sat is None:
            raise ValueError
        b_h = (1 - frac_sat) * bias_model(Planck18, M, a) + \
            frac_sat * bias_model(Planck18, M_host, a)
    else:
        b_h = bias_model(Planck18, M, a)

    # Matter density in unit M_sun / (comoving Mpc)^3 at redshift z_lens
    rho_m = _Omega_m * _rho_crit0

    # Linear matter correlation function
    k = np.logspace(-4, 4, 2000)
    P_lin = ccl.linear_matter_power(Planck18, k, a)  # (comoving Mpc)^3
    r_3d = np.logspace(-2, 2, 500)

    kr = np.outer(k, r_3d)
    integrand = k[:, None]**2 * P_lin[:, None] * np.sin(kr) / kr
    xi_lin = np.trapezoid(integrand, k, axis=0) / (2 * np.pi**2)   # dimensionless

    # Tinker 2005 scale-dependent bias correction
    eta = (1 + 1.17 * xi_lin)**1.49 / (1 + 0.69 * xi_lin)**2.09

    # Galaxy-matter correlation
    xi_gm = b_h * xi_lin * eta

    # Create a log-space linear interpolator
    xi_interp = interp1d(
        np.log(r_3d), xi_gm,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Abel projection Σ(R) via t-substitution (no singularity)
    rp_out = np.logspace(-2, np.log10(100), 60)  # comoving Mpc/h
    rp_Mpc = rp_out / h  # comoving Mpc

    rp_fine = np.logspace(-4, np.log10(100), 150)
    rp_fine_Mpc = rp_fine / h   # comoving Mpc

    t_max = 100.0
    t_grid = np.logspace(-4, np.log10(t_max), 150)

    Sigma_fine = np.zeros(len(rp_fine_Mpc))
    for i, R in enumerate(rp_fine_Mpc):
        r = np.sqrt(t_grid**2 + R**2)
        mask = r <= r_3d[-1]
        Sigma_fine[i] = 2 * rho_m * np.trapezoid(
            xi_interp(np.log(r[mask])), t_grid[mask])

    # Σ̄(<R) on fine grid
    R_Sigma_fine = rp_fine_Mpc * Sigma_fine
    I_cum = np.zeros_like(R_Sigma_fine)
    dr = np.diff(rp_fine_Mpc)
    I_cum[1:] = 0.5 * np.cumsum(dr * (R_Sigma_fine[1:] + R_Sigma_fine[:-1]))
    Sigma_bar_fine = 2 * I_cum / rp_fine_Mpc ** 2
    Sigma_bar_fine[0] = Sigma_fine[0]   # M_sun / (comoving Mpc)^2

    ds_fine = np.maximum(
        (Sigma_bar_fine - Sigma_fine) / 1.e12 / h, 0.0)
    ds_2h = np.interp(rp_Mpc, rp_fine_Mpc, ds_fine)  # h M_sun / (comoving pc)^2

    return rp_out, ds_2h


def _c_DM14(M, z_lens):
    """Dutton & Macciò (2014) concentration-mass relation.

    Args:
        M: Halo mass M_200c in M_sun.
        z_lens: Lens redshift.
    """
    a = lambda z: 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = lambda z: -0.101 + 0.026 * z
    log10_c = a(z_lens) + b(z_lens) * np.log10(M * h / 1.e12)
    return 10**log10_c