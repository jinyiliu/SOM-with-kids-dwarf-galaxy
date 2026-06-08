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

k_vec = np.logspace(start=-4, stop=4, num=100)

class DSigmaModel:
    def __init__(
            self,
            zl: np.ndarray | float,
            dsigma_mean_rp: np.ndarray | None=None,
            min_logmstar: np.ndarray | None=None,
            max_logmstar: np.ndarray | None=None,
    ):
        if isinstance(zl, float):
            zl = np.array([zl])

        if not len(min_logmstar) == len(max_logmstar) == len(zl):
            raise ValueError(
                "zl, min_logmstar, and max_logmstar must have the same length."
            )


        if dsigma_mean_rp is not None:
            if dsigma_mean_rp.ndim == 1:
                dsigma_mean_rp = np.tile(
                    dsigma_mean_rp[None, :],
                    reps=(len(zl), 1),
                )
        self.dsigma_mean_rp = dsigma_mean_rp

        self.zl = zl

        self.hmf = dict(
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
            delta_c=1.686,  # Critical density threshold for collapse
        )
        self.hod_params = dict(
            log10_obs_norm_c=10.521,
            log10_m_ch=11.145,
            g1=7.385,
            g2=0.201,
            sigma_log10_O_c=0.159,
            norm_s=0.562,
            pivot=13.0,
            alpha_s=-0.847,
            beta_s=2,
            b0=0.120,
            b1=1.177,
            b2=0.0,
            A_cen=None, # Assembly bias
            A_sat=None, # Assembly bias
        )
        self.hod_settings = dict(
            observables_file=None,
            # TODO: support input of stellar mass distribution
            obs_min=min_logmstar,
            obs_max=max_logmstar,
            zmin=np.array([0.0, 0.0]),
            zmax=np.array([0.5, 0.5]),
            nz=15,
            nobs=300,
            observable_h_unit="1/h^2",
        )
        self.spectra = Spectra(
            nonlinear_mode=None,
            dewiggle=False,
            response=False,
            pointmass=False,
            # If True, will be able to use Spectra.obs method
            compute_observable=False,
            mb=13.87, # Gas distribution mass pivot parameter
            # Compute beta_nl on the fly when nonlinear_mode is "bnl"
            beta_nl=None,
            one_halo_ktrunc=0.1,
            two_halo_ktrunc=2.0,
            poisson_model="constant",
            poisson_params={
                "poisson": 0.417,
            },
            hod_model="Cacciato",
            hod_params=self.hod_params,
            hod_settings=self.hod_settings,
            hod_settings_mm=None,
            obs_settings=None,
            # Effective parameter for nonlinear_model "fortuna"
            t_eff=0.,
            one_halo_ktrunc_ia=4.0,
            two_halo_ktrunc_ia=6.0,
            align_params=None,  # For SatelliteAlignment class

            # Kwargs for the parent class of Spectra: HaloModelIngredients
            **self.hmf,
            # Kwargs for parent class of HaloModelIngredients: CosmologyBase
            z_vec=self.zl,
            **cosmo_Planck18,
        )

    def get_predict_func(
            self,
            param_names: list[str],
            evaluate_at_mean_rp: bool=True,
    ) -> Callable:
        if evaluate_at_mean_rp: # Generate data vector for MCMC
            if self.dsigma_mean_rp is None:
                raise ValueError(
                    "dsigma_mean_rp must have been set."
                )
            def predict_func(param_values: np.ndarray) -> np.ndarray | list[np.ndarray]:
                self._update_spectra(param_names, param_values)
                _sep, _dsigma = Pgm2DSigma(
                    model=self.spectra,
                    rpmin=0.01,
                    rpmax=40,
                    components=False,
                )
                dsigma = []
                for mean_rp, _ds in zip(self.dsigma_mean_rp, _dsigma):
                    dsigma.append(
                        np.interp(mean_rp, _sep, _ds)
                    )

                return np.array(dsigma).flatten()
        else: # Generate dsigma curve for visualisation
            def predict_func(param_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                self._update_spectra(param_names, param_values)
                sep, dsigma = Pgm2DSigma(
                    model=self.spectra,
                    rpmin=0.01,
                    rpmax=40,
                    components=False,
                )
                return sep, dsigma
        return predict_func

    def _update_spectra(
            self, param_names: list[str], param_values: np.ndarray,
    ):
        if not len(param_values) == len(param_names):
            raise ValueError
        for param_name, param_values in zip(param_names, param_values):
            if param_name in self.hmf.keys():
                self.hmf[param_name] = param_values
                self.spectra.update(**{param_name: param_values})
            elif param_name in self.hod_params.keys():
                self.hod_params[param_name] = param_values
                self.spectra.update(
                    hod_params = self.hod_params
                )
            else:
                raise NotImplementedError


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