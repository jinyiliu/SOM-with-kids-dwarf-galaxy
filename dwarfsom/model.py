import os
import pickle as pk
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

h = astropy_Planck18.H0.value / 100
_satellite_HOD_log10_M0 = 7.0
_satellite_HOD_log10_M1 = 13.3
_satellite_HOD_alpha = 1.0

_cache_dir = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL"

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


def DSigmaModel(
        rp: np.ndarray,
        z_lens: float,
        log10_M: float,
        f_c: float,
        frac_sat: float,
        log10_M_star: float,
        tau: float | None=None,
        return_components: bool=False,
):
    ds_1h_cm = DSigma_1h_cm(rp, log10_M, z_lens, f_c=f_c)
    ds_1h_sm_sub = DSigma_1h_sm_sub(rp, log10_M, z_lens, f_c=1., tau=tau)
    ds_1h_sm_host = DSigma_1h_sm_host(rp, z_lens, f_c=1.)
    ds_2h = DSigma_2h(rp, log10_M, z_lens)
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
    rp_grid, ds_grid = compute_host_dsigma(z_lens, c, f_c)
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
    rp_grid, ds_grid = precompute_2h(z_lens, log10_M)
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
    return np.interp(rp, r_fine, Sigma_fine)


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

    Args:
        rp_sat: Satellite distance to host halo centre in comoving Mpc.
        log10_M: log10 of host halo mass in M_sun.
        z_lens: Lens redshift.
        c:
        f_c:
    """
    nfw, a, M = _get_nfw_profile(
        log10_M, z_lens, c, f_c, truncated=False, analytic=True)

    r_Mpc = np.atleast_1d(np.asarray(rp_sat, dtype=float))  # comoving Mpc

    Sigma = nfw.projected(Planck18, r_Mpc, M, a)
    P = 2 * np.pi * r_Mpc * Sigma
    if len(r_Mpc) > 1:
        P = P / np.trapezoid(P, r_Mpc)
    return P


def compute_host_dsigma(z_lens, c, f_c):
    """Precompute the host halo ESD contribution for given z_lens."""
    if len(_cache_host_terms) == 0:
        _cache_host_terms.update(
            precompute_host_dsigma_terms(
                z_lens, c, f_c, n_rs=70
            )
        )

    log10_M_mid = _cache_host_terms["log10_M_mid"]
    dlog10_M = _cache_host_terms["dlog10_M"]
    dndlogM = _cache_host_terms["dndlogM"]
    rs_grid_M = _cache_host_terms["rs_grid_M"]
    Sigma_M_rs = _cache_host_terms["Sigma_M_rs"]
    M_mid = 10 ** log10_M_mid

    N_sat = _satellite_HOD(
        log10_M_mid,
        log10_M0=_satellite_HOD_log10_M0,
        log10_M1=_satellite_HOD_log10_M1,
        alpha=_satellite_HOD_alpha,
    )

    n_bar = np.trapezoid(dndlogM * N_sat, log10_M_mid)

    if n_bar == 0:
        rp_out_Mpc = np.logspace(-5, 2, 200)
        return rp_out_Mpc, np.zeros(len(rp_out_Mpc))

    rp_out_Mpc = np.logspace(-5, 2, 200)    # comoving Mpc
    result_Sigma = np.zeros(len(rp_out_Mpc))

    # Loop over halo mass
    for i, lm in enumerate(log10_M_mid):
        if N_sat[i] == 0:
            continue

        P_rs = _satellite_radial_distribution(
            rs_grid_M[i], lm, z_lens, c=c, f_c=f_c,
        )   # shape (n_rs,)
        Sigma_rs = Sigma_M_rs[i] # shape (n_rs, len(rp_out))

        result_Sigma += dndlogM[i] * N_sat[i] * np.trapezoid(
            P_rs[:, None] * Sigma_rs, rs_grid_M[i], axis=0) * dlog10_M

    result_Sigma /= n_bar

    rp_fine_Mpc = np.logspace(-5, 2, 500)  # comoving Mpc
    Sigma_fine = np.interp(  # h M_sun / (comoving pc)^2
        np.log(rp_fine_Mpc), np.log(rp_out_Mpc), result_Sigma)

    rp_fine = rp_fine_Mpc * h  # comoving Mpc/h

    R_Sigma = rp_fine * Sigma_fine
    I_cum = np.zeros_like(R_Sigma)
    dr = np.diff(rp_fine)
    I_cum[1:] = 0.5 * np.cumsum(dr * (R_Sigma[1:] + R_Sigma[:-1]))
    Sigma_bar = 2.0 * I_cum / rp_fine ** 2
    Sigma_bar[0] = Sigma_fine[0]

    ds_fine = np.maximum((Sigma_bar - Sigma_fine), 0.0)

    rp_out = rp_out_Mpc * h  # comoving Mpc/h
    ds_pop = np.interp(rp_out, rp_fine, ds_fine)
    return rp_out, ds_pop


def precompute_host_dsigma_terms(
        z_lens, c, f_c, n_rs: int=70,
):
    from pyccl.halos import MassFuncTinker08

    a = 1.0 / (1.0 + z_lens)
    rp_out_Mpc = np.logspace(-5, 2, 200)  # comoving Mpc

    log10_M = np.linspace(_satellite_HOD_log10_M0, 16.0, 40)
    log10_M_mid = 0.5 * (log10_M[1:] + log10_M[:-1])
    dlog10_M = log10_M[1] - log10_M[0]

    M_mid = 10 ** log10_M_mid
    hmf = MassFuncTinker08(mass_def=MassDef200m)
    dndlogM = hmf(Planck18, M_mid, a)

    Sigma_M_rs = np.zeros(shape=(
        len(M_mid), n_rs, len(rp_out_Mpc)
    ))
    rs_grid_M = np.zeros(shape=(len(M_mid), n_rs))

    # Loop over halo mass
    for i, lm in enumerate(log10_M_mid):
        r_vir = MassDef200m.get_radius(Planck18, M_mid[i], a)  # physical Mpc
        r_vir = r_vir / a  # comoving Mpc

        rs_grid_M[i] = np.logspace(
            -3, np.log10(0.95 * r_vir), n_rs,
        )  # comoving Mpc

        for j, rs in enumerate(rs_grid_M[i]):
            Sigma = _Sigma_NFW_offset(
                rp=rp_out_Mpc * h,  # comoving Mpc/h
                rp_sat=rs * h,  # comoving Mpc/h
                log10_M=lm,
                z_lens=z_lens,
                c=c,
                f_c=f_c,
            )  # h M_sun / (comoving pc)^2
            Sigma_M_rs[i, j] = Sigma

    return {
        "log10_M_mid": log10_M_mid,
        "dlog10_M": dlog10_M,
        "dndlogM": dndlogM,
        "rs_grid_M": rs_grid_M,
        "Sigma_M_rs": Sigma_M_rs,
    }


def precompute_2h(z_lens, log10_M):
    """Precompute the 2-halo ESD for a given (z_lens, log10_M)."""
    from pyccl.halos import HaloBiasTinker10

    a = 1.0 / (1.0 + z_lens)
    M = 10 ** log10_M

    bias_model = HaloBiasTinker10(mass_def=MassDef200m)
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
    xi_gm = b_h * eta * xi_lin

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