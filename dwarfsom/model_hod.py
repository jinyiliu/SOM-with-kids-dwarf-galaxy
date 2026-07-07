import os

n_threads = 4
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)

import copy
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from onepower import Spectra

from numpy.typing import ArrayLike
from typing import Callable

from ._pk2real import PkTransformer

cosmo_Planck18 = dict(
    h0=0.711,
    omega_c=0.287,
    omega_b=0.01,
    n_s=0.978,
    sigma_8=0.801,
)

h0_GAMA = 0.7 # See Driver et al. (2022)

k_vec = np.logspace(start=-4, stop=4, num=100)

_hmf = dict(
    k_vec=k_vec,
    # Keyword arguments for hmf
    lnk_min=np.log(10 ** -4),
    lnk_max=np.log(10 ** 4),
    dlnk=(np.log(10 ** 4) - np.log(10 ** (-4))) / 100,
    Mmin=9.,
    Mmax=16.,
    dlog10m=0.05,
    mdef_model="SOMean",
    hmf_model="Tinker10",
    transfer_model="CAMB",
    transfer_params=None,
    growth_model="CambGrowth",
    growth_params=None,
    # Keyword arguments for halomod
    bias_model="Tinker10",
    halo_profile_model_dm="NFW",
    halo_profile_model_sat="NFW",
    halo_concentration_model_dm="Duffy08",
    halo_concentration_model_sat="Duffy08",
    norm_cen=1., # Normalisation of c(M) relation for central galaxies
    norm_sat=1., # Normalisation of c(M) relation for satellite galaxies
    eta_cen=0., # Bloating parameter for central galaxies
    eta_sat=0., # Bloating parameter for satellite galaxies
    overdensity=200,
    delta_c=1.686, # Critical density threshold for collapse
)

_hod_params = dict(
    log10_obs_norm_c=10.521,
    log10_m_ch=11.145,
    g1=7.385,
    g2=0.201,
    sigma_log10_O_c=0.159,
    norm_s=0.562,
    pivot=13.,
    alpha_s=-1.,
    beta_s=2,
    b0=0.120,
    b1=0.,
    b2=0.,
    A_cen=None, # Assembly bias
    A_sat=None, # Assembly bias
)


class DSigmaModel:
    def __init__(
            self,
            zl: np.ndarray | float,
            param_names: list[str],
            dsigma_mean_rp: np.ndarray | None=None,
            min_logmstar: np.ndarray | None=None,
            max_logmstar: np.ndarray | None=None,
            dndlogmstar: np.ndarray | None=None,
            use_single_power_law_shmr: bool=False,
    ):
        """

        Args:
            zl: Redshifts of the lenses.
            param_names: Parameter names.
            dsigma_mean_rp:
            min_logmstar: In unit of solar mass without little h.
            max_logmstar: In unit of solar mass without little h.
            dndlogmstar: Stellar mass distribution in the stellar mass range
                (min_logmstar, max_logmstar). It should be normalised such that
                np.sum(dndlogmstar[0]) == len(dndlogmstar[0]).
        """
        if isinstance(zl, float):
            zl = np.array([zl])

        if not len(min_logmstar) == len(max_logmstar) == len(zl):
            raise ValueError(
                "zl, min_logmstar, and max_logmstar must have the same length."
            )
        self.min_logmstar = min_logmstar + 2 * np.log10(h0_GAMA)
        self.max_logmstar = max_logmstar + 2 * np.log10(h0_GAMA)
        if dndlogmstar is not None:
            if not len(dndlogmstar) == len(min_logmstar) == len(max_logmstar):
                raise ValueError(
                    "dndlogmstar must have the same length as min_logmstar and max_logmstar."
                )
        self.dndlogmstar = dndlogmstar
        self._nobs = dndlogmstar.shape[1] if dndlogmstar is not None else 300
        self.use_single_power_law_shmr = use_single_power_law_shmr

        if dsigma_mean_rp is not None:
            if dsigma_mean_rp.ndim == 1:
                dsigma_mean_rp = np.tile(
                    dsigma_mean_rp[None, :],
                    reps=(len(zl), 1),
                )
        self.dsigma_mean_rp = dsigma_mean_rp

        self.zl = zl
        self.param_names = param_names

        self.hod_settings = dict(
            observables_file=None,
            obs_min=self.min_logmstar,
            obs_max=self.max_logmstar,
            dndlogmstar=self.dndlogmstar,
            zmin=np.array([0.0, 0.0]),
            zmax=np.array([0.5, 0.5]),
            nz=15,
            nobs=self._nobs,
            observable_h_unit="1/h^2",
        )

    def model_evaluated_at_mean_rp(self, param_values) -> np.ndarray:
        _sep, _dsigma = self.model(param_values)
        dsigma = []
        for mean_rp, _ds in zip(self.dsigma_mean_rp, _dsigma):
            dsigma.append(
                np.interp(mean_rp, _sep, _ds)
            )
        return np.array(dsigma).flatten()


    def model(self, param_values) -> tuple[np.ndarray, np.ndarray]:
        hmf, hod_params = _update_hmf_hod_params(
            self.param_names,
            param_values,
            self.use_single_power_law_shmr,
        )
        spectra = _create_OnePowerSpectra_instance(
            self.zl, hmf, hod_params, self.hod_settings)
        _sep, _dsigma = Pgm2DSigma(
            model=spectra,
            rpmin=0.01,
            rpmax=40,
            components=False,
        )
        return _sep, _dsigma



def Pgm2DSigma(
        model: Spectra,
        rpmin=None,
        rpmax=None,
        components: bool=False,
):
    transformer = PkTransformer(
        corr_type="ds",
        model=model,
        k_min=None,
        k_max=None,
        sep_min_in=rpmin,
        sep_max_in=rpmax,
        n_transform=None,
        z_s=None,
        components=components,
    )
    return transformer()


def _update_hmf_hod_params(
        param_names: list[str],
        param_values: np.ndarray,
        use_single_power_law_shmr: bool
) -> tuple[dict, dict]:
    hmf = copy.deepcopy(_hmf)
    hod_params = copy.deepcopy(_hod_params)
    if use_single_power_law_shmr:
        hod_params["g2"] = "g1"
    if not len(param_values) == len(param_names):
        raise ValueError
    for param_name, param_value in zip(param_names, param_values):
        if param_name in hmf.keys():
            hmf[param_name] = param_value
        elif param_name in hod_params.keys():
            hod_params[param_name] = param_value
        else:
            raise NotImplementedError(
                f"Parameter {param_name} is not implemented."
            )
    return hmf, hod_params


def _create_OnePowerSpectra_instance(
        zl: np.ndarray,
        hmf: dict,
        hod_params: dict,
        hod_settings: dict,
) -> Spectra:
    spectra = Spectra(
        nonlinear_mode=None,
        dewiggle=False,
        response=False,
        pointmass=False,
        # If True, will be able to use Spectra.obs method
        compute_observable=False,
        mb=13.87,  # Gas distribution mass pivot parameter
        # Compute beta_nl on the fly when nonlinear_mode is "bnl"
        beta_nl=None,
        one_halo_ktrunc=0.1,
        two_halo_ktrunc=2.0,
        poisson_model="constant",
        poisson_params={
            "poisson": 0.417,
        },
        hod_model="Cacciato",
        hod_params=hod_params,
        hod_settings=hod_settings,
        hod_settings_mm=None,
        obs_settings=None,
        # Effective parameter for nonlinear_model "fortuna"
        t_eff=0.,
        one_halo_ktrunc_ia=4.0,
        two_halo_ktrunc_ia=6.0,
        align_params=None,  # For SatelliteAlignment class

        # Kwargs for the parent class of Spectra: HaloModelIngredients
        **hmf,
        # Kwargs for parent class of HaloModelIngredients: CosmologyBase
        z_vec=zl,
        **cosmo_Planck18,
    )
    return spectra

