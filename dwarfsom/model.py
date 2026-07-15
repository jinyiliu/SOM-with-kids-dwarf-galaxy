import numpy as np
import pyccl as ccl
from pyccl.halos import (
    HaloProfileNFW,
    ConcentrationConstant,
    ConcentrationDuffy08,
    MassDef200m,
)
from astropy.cosmology import Planck18 as astropy_Planck18
from scipy.interpolate import interp1d

h = astropy_Planck18.H0.value / 100
_satellite_HOD_log10_M0 = 7.0
_satellite_HOD_log10_M1 = 13.3
_satellite_HOD_alpha = 1.0

_cache_host_dsigma = {}
_cache_2h = {}
_Omega_m = astropy_Planck18.Om0
_rho_crit0 = 2.775e11 * h**2  # M_sun/(physical Mpc)^3

Planck18 = ccl.Cosmology(
    Omega_c=(astropy_Planck18.Om0 - astropy_Planck18.Ob0),
    Omega_b=astropy_Planck18.Ob0,
    h=h,
    T_CMB=astropy_Planck18.Tcmb0.value,
    Neff=astropy_Planck18.Neff,
    n_s=0.9649,
    sigma8=0.8111,
)


def DSigma(
        rp: np.ndarray,
        z_lens: float,
        log10_M: float,
        f_c: float,
        frac_sat: float,
        tau: float | None=None,
):
    ds_1h_cm = DSigma_1h_cm(rp, log10_M, z_lens, f_c=f_c)
    ds_1h_sm_sub = DSigma_1h_sm_sub(rp, log10_M, z_lens, f_c=f_c, tau=tau)
    ds_1h_sm_host = DSigma_1h_sm_host(rp, z_lens, f_c=f_c)
    ds_2h = DSigma_2h(rp, log10_M, z_lens)
    ds = (
            (1 - frac_sat) * ds_1h_cm +
            frac_sat * (ds_1h_sm_sub + ds_1h_sm_host) +
            ds_2h
    )
    return ds


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
        tau: float | None=None,
):
    """BMO smoothly-truncated NFW subhalo ESD (Baltz et al. 2009).

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun.
        z_lens: Lens redshift.
        c: Concentration. Mutually exclusive with f_c.
        f_c: Duffy08 amplitude. Mutually exclusive with c.
        tau: r_t / r_s (dimensionless truncation). Default: 2.5.
    """
    if (c is None) == (f_c is None):
        raise ValueError("Provide exactly one of: c or f_c.")

    a = 1.0 / (1.0 + z_lens)
    M = 10.0 ** log10_M
    r_vir = MassDef200m.get_radius(Planck18, M, a) / a  # in comoving physical Mpc

    if c is not None:
        conc_val = c
    else:
        conc = ConcentrationDuffy08(fc_bar=f_c, mass_def=MassDef200m)
        conc_val = float(conc(Planck18, M, a))

    r_s = r_vir / conc_val

    if tau is None:
        tau_val = 2.5  # default: r_t = 2.5 * r_s
    else:
        tau_val = tau

    M0 = M / (np.log(1 + conc_val) - conc_val / (1 + conc_val))

    rp_phys = np.atleast_1d(np.asarray(rp, dtype=float)) / h

    r_min = max(np.min(rp_phys) * 0.3, 1e-5)
    r_max = np.max(rp_phys) * 1.5
    r_fine = np.logspace(np.log10(r_min), np.log10(r_max), 200)

    Sigma_fine = _Sigma_BMO(r_fine, M0, r_s, tau_val)
    rS = r_fine * Sigma_fine
    I_cum = np.zeros_like(rS)
    dr = np.diff(r_fine)
    I_cum[1:] = 0.5 * np.cumsum(dr * (rS[1:] + rS[:-1]))

    r_sub = np.logspace(
        np.log10(max(r_s * 1e-3, 1e-5)),
        np.log10(r_fine[0]), 100)
    Sigma_sub = _Sigma_BMO(r_sub, M0, r_s, tau_val)
    I_0 = np.trapezoid(r_sub * Sigma_sub, r_sub)

    I_cum += I_0
    Sigma_bar = 2.0 * I_cum / r_fine**2
    Sigma_bar[0] = 2.0 * I_0 / r_fine[0]**2

    ds_fine = (Sigma_bar - Sigma_fine) / 1e12
    return np.interp(rp_phys, r_fine, np.maximum(ds_fine, 0.0))


def DSigma_1h_sm_host(
        rp: np.ndarray,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
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
    """
    key = (z_lens, c, f_c)
    if key not in _cache_host_dsigma:
        _cache_host_dsigma[key] = _precompute_host_dsigma(
            z_lens, c, f_c)
    rp_grid, ds_grid = _cache_host_dsigma[key]
    return np.interp(np.atleast_1d(rp), rp_grid, ds_grid)


def DSigma_2h(
        rp: np.ndarray,
        log10_M: float,
        z_lens: float,
):
    """Two-halo ESD from linear bias with Tinker 2005 scale dependence.

    ξ_gm(r) = b(M) × η(r) × ξ_lin(r)

    The projected surface density Σ(R) is obtained via Abel
    integral, then ΔΣ_2h = Σ̄(<R) − Σ(R).

    Args:
        rp: Projected separations in Mpc/h.
        log10_M: log10 of halo mass M_200m in M_sun.
        z_lens: Lens redshift.
    """
    key = (z_lens, log10_M)
    if key not in _cache_2h:
        _cache_2h[key] = _precompute_2h(z_lens, log10_M)
    rp_grid, ds_grid = _cache_2h[key]
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
        R: Projected radii in physical Mpc.
        M0: Profile normalisation mass in M_sun.
        r_s: Scale radius (same units as R).
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

    return pref * (term1 + term2 + term3 + term4)


def _satellite_HOD(
        log10_M: float | np.ndarray,
        log10_M0: float=_satellite_HOD_log10_M0,
        log10_M1: float=_satellite_HOD_log10_M1,
        alpha: float=_satellite_HOD_alpha,
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


def _satellite_radial_distribution(
        rp_sat: np.ndarray,
        log10_M: float,
        z_lens: float,
        c: float | None=None,
        f_c: float | None=None,
):
    """Projected satellite number density profile.

    Satellites trace the host NFW projected mass:
    P(r_p_sat | M) is propotional to 2π rp_sat x Σ_NFW(rp_sat | M)
    """
    nfw, a, M = _get_nfw_profile(
        log10_M, z_lens, c, f_c, truncated=False, analytic=True)

    r_phys = np.atleast_1d(np.asarray(rp_sat, dtype=float)) / h # Mpc/h -> physical Mpc
    Sigma = nfw.projected(Planck18, r_phys, M, a)
    P_phys = 2 * np.pi * r_phys * Sigma
    if len(r_phys) > 1:
        P_phys = P_phys / np.trapezoid(P_phys, r_phys)
    return P_phys / h


def _precompute_host_dsigma(z_lens, c, f_c):
    """Precompute the host halo ESD contribution for given z_lens."""
    from pyccl.halos import MassFuncTinker08

    a = 1.0 / (1.0 + z_lens)

    log10_M = np.linspace(_satellite_HOD_log10_M0, 16.0, 40)
    log10_M_mid = 0.5 * (log10_M[1:] + log10_M[:-1])
    dlogM = log10_M[1] - log10_M[0]

    M_mid = 10 ** log10_M_mid
    N_sat = _satellite_HOD(
        log10_M_mid,
        log10_M0=_satellite_HOD_log10_M0,
        log10_M1=_satellite_HOD_log10_M1,
        alpha=_satellite_HOD_alpha,
    )

    hmf = MassFuncTinker08(mass_def=MassDef200m)
    dndlogM = hmf(Planck18, M_mid, a)
    n_bar = np.trapezoid(dndlogM * N_sat, log10_M_mid)

    if n_bar == 0:
        rp_out = np.logspace(-5, 2, 200)
        return rp_out, np.zeros(len(rp_out))

    rp_out = np.logspace(-5, 2, 200)
    n_rs = 25
    n_phi = 60

    phi = np.linspace(0, 2 * np.pi, n_phi)
    cos_phi = np.cos(phi)

    result_Sigma = np.zeros(len(rp_out))

    for i, lm in enumerate(log10_M_mid):
        if N_sat[i] == 0:
            continue

        r_vir = MassDef200m.get_radius(Planck18, M_mid[i], a) / h
        rs_grid = np.logspace(
            np.log10(0.001), np.log10(0.95 * r_vir), n_rs,
        )
        P_rs = _satellite_radial_distribution(
            rs_grid, lm, z_lens, c=c, f_c=f_c,
        )

        nfw, _, M = _get_nfw_profile(
            lm, z_lens, c, f_c, truncated=False, analytic=True,
        )

        r_max = max(np.max(rp_out), rs_grid[-1] + np.max(rp_out))
        R_dense = np.logspace(np.log10(1e-4), np.log10(r_max), 500) / h
        Sigma_dense = nfw.projected(Planck18, R_dense, M, a)

        Sigma_rs = np.zeros((n_rs, len(rp_out)))

        for j, rs in enumerate(rs_grid):
            rs_phys = rs / h
            rp_phys = rp_out / h

            rs2 = rs_phys ** 2
            r2 = rp_phys[:, None] ** 2
            d2 = rs2 + r2 + 2 * rs_phys * rp_phys[:, None] * cos_phi[None, :]
            d = np.sqrt(d2)

            Sigma_phi_rp = np.mean(
                np.interp(d.ravel(), R_dense, Sigma_dense).reshape(d.shape),
                axis=1,
            )
            Sigma_rs[j] = P_rs[j] * Sigma_phi_rp

        result_Sigma += dndlogM[i] * N_sat[i] * np.trapezoid(
            Sigma_rs, rs_grid, axis=0) * dlogM

    result_Sigma /= n_bar

    rp_fine = np.logspace(-5, 2, 200)
    Sigma_fine = np.interp(
        np.log(rp_fine), np.log(rp_out), result_Sigma)

    rp_fine_phys = rp_fine / h
    R_Sigma = rp_fine_phys * Sigma_fine
    I_cum = np.zeros_like(R_Sigma)
    dr = np.diff(rp_fine_phys)
    I_cum[1:] = 0.5 * np.cumsum(dr * (R_Sigma[1:] + R_Sigma[:-1]))
    Sigma_bar = 2.0 * I_cum / rp_fine_phys ** 2
    Sigma_bar[0] = Sigma_fine[0]

    ds_fine = np.maximum((Sigma_bar - Sigma_fine) / 1e12, 0.0)
    rp_out_phys = rp_out / h
    ds_pop = np.interp(rp_out_phys, rp_fine_phys, ds_fine)
    return rp_out, ds_pop


def _precompute_2h(z_lens, log10_M):
    """Precompute the 2-halo ESD for a given (z_lens, M_h)."""
    from pyccl.halos import HaloBiasTinker10

    a = 1.0 / (1.0 + z_lens)
    M = 10 ** log10_M

    bias_model = HaloBiasTinker10(mass_def=MassDef200m)
    b_h = bias_model(Planck18, M, a)
    rho_m = _Omega_m * _rho_crit0 / a**3

    # Linear matter correlation function
    k = np.logspace(-4, 4, 2000)
    P_lin = ccl.linear_matter_power(Planck18, k, a)
    r_3d = np.logspace(-2, 2, 500)

    kr = np.outer(k, r_3d)
    integrand = k[:, None]**2 * P_lin[:, None] * np.sin(kr) / kr
    xi_lin = np.trapezoid(integrand, k, axis=0) / (2 * np.pi**2)

    # Tinker 2005 scale-dependent bias correction
    eta = (1 + 1.17 * xi_lin)**1.49 / (1 + 0.69 * xi_lin)**2.09

    # Galaxy-matter correlation
    xi_gm = b_h * eta * xi_lin

    xi_interp = interp1d(
        np.log(r_3d), xi_gm, kind='linear',
        bounds_error=False, fill_value=0.0,
    )

    # Abel projection Σ(R) via t-substitution (no singularity)
    rp_out = np.logspace(-2, np.log10(100), 60)
    rp_phys = rp_out / h

    rp_fine = np.logspace(-4, np.log10(100), 150)
    rp_fine_phys = rp_fine / h

    t_max = 100.0
    t_grid = np.logspace(-4, np.log10(t_max), 150)

    Sigma_fine = np.zeros(len(rp_fine_phys))
    for i, R in enumerate(rp_fine_phys):
        r = np.sqrt(t_grid**2 + R**2)
        mask = r <= r_3d[-1]
        Sigma_fine[i] = 2 * rho_m * np.trapezoid(
            xi_interp(np.log(r[mask])), t_grid[mask])

    # Σ̄(<R) on fine grid
    R_Sigma_fine = rp_fine_phys * Sigma_fine
    I_cum = np.zeros_like(R_Sigma_fine)
    dr = np.diff(rp_fine_phys)
    I_cum[1:] = 0.5 * np.cumsum(dr * (R_Sigma_fine[1:] + R_Sigma_fine[:-1]))
    Sigma_bar_fine = 2 * I_cum / rp_fine_phys**2
    Sigma_bar_fine[0] = Sigma_fine[0]

    ds_fine = np.maximum(
        (Sigma_bar_fine - Sigma_fine) / 1e12, 0.0)
    ds_2h = np.interp(rp_phys, rp_fine_phys, ds_fine)

    return rp_out, ds_2h