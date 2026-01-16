import numpy as np

from dataclasses import dataclass
from astropy.cosmology import FlatLambdaCDM
from treecorr import NGCorrelation, Catalog
from dsigma.physics import critical_surface_density

cosmo_default = FlatLambdaCDM(H0=100, Om0=0.3)

@dataclass
class Lens:
    ra: np.ndarray
    dec: np.ndarray
    dndz: tuple[np.ndarray, np.ndarray]
    dndmstar: tuple[np.ndarray, np.ndarray]
    w: np.ndarray | None = None

@dataclass
class Source:
    ra: np.ndarray
    dec: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    w: np.ndarray
    dndz: tuple[np.ndarray, np.ndarray]
    m: float = 0. # multiplicative bias

@dataclass
class Random:
    ra: np.ndarray
    dec: np.ndarray
    w: np.ndarray | None = None


class DSigma:
    def __init__(
            self,
            lens: Lens,
            source: Source,
            random: Random | None=None,
            cosmo=cosmo_default,
            n_rp_bins: int=8,
            min_rp: float=0.01,
            max_rp: float=10.0,
    ):
        """Excess Surface Density Estimator using TreeCorr."""
        self.lens = lens
        self.source = source
        self.random = random
        self.cosmo = cosmo
        self.n_rp_bins = n_rp_bins
        self.min_rp = min_rp
        self.max_rp = max_rp

        self._effective_sigma_crit = self.effective_critical_surface_density()

        lens = Catalog(
            ra=lens.ra, ra_units="degrees",
            dec=lens.dec, dec_units="degrees",
            w=self.lens.w,
        )
        source = Catalog(
            ra=self.source.ra, ra_units="degrees",
            dec=self.source.dec, dec_units="degrees",
            g1=self.source.e1, g2=self.source.e2,
            w=self.source.w,
        )
        ng = NGCorrelation(
            min_sep=self.hMpc2degree(self.min_rp),
            max_sep=self.hMpc2degree(self.max_rp),
            nbins=self.n_rp_bins,
            sep_units="degree",
        )
        ng.process(lens, source)

        self.dsigma_tangential = ng.xi * self._effective_sigma_crit
        self.dsigma_cross = ng.xi_im * self._effective_sigma_crit
        self.cov = ng.cov * self._effective_sigma_crit**2
        self.mean_rp = self.degree2hMpc(ng.meanr)

        if self.random is not None:
            self.boost = ...
            self.dsigma_rand = ...
            self.dsigma_tangential *= self.boost
            self.dsigma_cross *= self.boost
            self.dsigma_tangential -= self.dsigma_rand[0]
            self.dsigma_cross -= self.dsigma_rand[1]

        # Multiplicative shear bias correction
        self.dsigma_tangential /= (1 + self.source.m)
        self.dsigma_cross /= (1 + self.source.m)


    def get_boost(self): # TODO
        assert self.random is not None

    def get_dsigma_rand(self): # TODO
        assert self.random is not None

    def effective_critical_surface_density(self):
        z_l, z_s = np.meshgrid(self.lens.dndz[0], self.source.dndz[0], indexing="ij")
        weights = np.outer(self.lens.dndz[1], self.source.dndz[1])
        sigma_crit = critical_surface_density(z_l, z_s, self.cosmo)
        finite_mask = np.isfinite(sigma_crit)
        sigma_crit_eff = np.average(
            sigma_crit[finite_mask],
            weights=weights[finite_mask],
        )
        return sigma_crit_eff

    def degree2hMpc(self, degree: np.ndarray | float):
        DA = np.average(
            a=self.cosmo.angular_diameter_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        )
        radian = degree * np.pi / 180.
        hMpc = radian * DA
        return hMpc

    def hMpc2degree(self, hMpc: np.ndarray | float):
        DA = np.average(
            a=self.cosmo.angular_diameter_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        )
        radian = hMpc / DA
        degree = radian * 180. / np.pi
        return degree