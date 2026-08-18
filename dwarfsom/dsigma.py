import os.path
import numpy as np
import pandas as pd

from glob import glob
from astropy.io import fits
from natsort import natsorted
from dataclasses import dataclass
from treecorr import NGCorrelation, Catalog
from astropy import units as u
from astropy.cosmology import Planck18
from astropy import constants as c
from astropy.cosmology import units as cu

from dwarfsom.build_catalogue import KiDS_RANDOMS_DIR

@dataclass
class Lens:
    ra: np.ndarray
    dec: np.ndarray
    dndz: tuple[np.ndarray, np.ndarray]
    dndlogmstar: tuple[np.ndarray, np.ndarray]
    w: np.ndarray | None = None

    def __eq__(self, other):
        if not isinstance(other, Lens):
            return False
        ret = (
            np.array_equal(self.ra, other.ra) and
            np.array_equal(self.dec, other.dec) and
            np.array_equal(self.dndz[0], other.dndz[0]) and
            np.array_equal(self.dndz[1], other.dndz[1]) and
            np.array_equal(self.dndlogmstar[0], other.dndlogmstar[0]) and
            np.array_equal(self.dndlogmstar[1], other.dndlogmstar[1]) and
            (
                (self.w is None and other.w is None) or
                np.array_equal(self.w, other.w)
            )
        )
        return ret



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

    @classmethod
    def from_random_catalogue(cls, savepath: str) -> "Random":
        if savepath.endswith(".fits"):
            with fits.open(savepath) as hdul:
                data = hdul[1].data
                ra = data["ALPHA_J2000"]
                dec = data["DELTA_J2000"]
            return cls(ra=ra, dec=dec)

        if savepath.endswith(".csv"):
            df = pd.read_csv(savepath)
            ra = df["ALPHA_J2000"].values
            dec = df["DELTA_J2000"].values
            return cls(ra=ra, dec=dec)


@dataclass
class DSigmaData:
    mean_rp: np.ndarray
    mean_rp_arcmin: np.ndarray
    npairs: np.ndarray
    dsigma_tangential: np.ndarray
    dsigma_cross: np.ndarray
    dsigma_stderr: np.ndarray
    boost: np.ndarray
    dsigma_rand_tangential: np.ndarray
    dsigma_rand_cross: np.ndarray
    boost_array: np.ndarray
    dsigma_rand_array: np.ndarray


    @classmethod
    def load_csv(
            cls,
            fname: str,
            save_dir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/",
    ) -> "DSigmaData":
        df = pd.read_csv(os.path.join(save_dir, fname))
        boost_array = df.filter(like="boost_randcat").values.T
        dsigma_rand_array = np.vstack((
            [df.filter(like="dsigma_rand_tangential_randcat").values.T],
            [df.filter(like="dsigma_rand_cross_randcat").values.T],
        ))
        return cls(
            mean_rp=df["mean_rp_Mpc_h_inv"].values,
            mean_rp_arcmin=df["mean_rp_arcmin"].values,
            npairs=df["npairs"].values,
            dsigma_tangential=df["dsigma_tangential"].values,
            dsigma_cross=df["dsigma_cross"].values,
            dsigma_stderr=df["dsigma_stderr"].values,
            boost=df["boost"].values,
            dsigma_rand_tangential=df["dsigma_rand_tangential"].values,
            dsigma_rand_cross=df["dsigma_rand_cross"].values,
            boost_array=boost_array,
            dsigma_rand_array=dsigma_rand_array,
        )


class DSigma:
    def __init__(
            self,
            lens: Lens,
            source: Source,
            randoms: list[Random] | None=None,
            cosmo=Planck18,
            n_rp_bins: int=15,
            min_rp: float=0.02, # comoving Mpc/h
            max_rp: float=20.0, # comoving Mpc/h
            patch_centers: str | None=None,
            npatch: int | None=None,
            var_method: str="shot",
    ):
        """Excess Surface Density Estimator using TreeCorr."""
        if var_method != "shot":
            if patch_centers is None and npatch is None:
                raise ValueError

        self.lens = lens
        self.source = source
        self.randoms = randoms
        self.cosmo = cosmo
        self.n_rp_bins = n_rp_bins
        self.min_rp = min_rp
        self.max_rp = max_rp
        self.patch_centers = patch_centers
        self.npatch = npatch

        self._effective_sigma_crit = self.effective_critical_surface_density()

        self._lens_Catalog = Catalog(
            ra=lens.ra, ra_units="degrees",
            dec=lens.dec, dec_units="degrees",
            w=self.lens.w,
            patch_centers=patch_centers,
            npatch=npatch,
        )
        self._source_Catalog = Catalog(
            ra=self.source.ra, ra_units="degrees",
            dec=self.source.dec, dec_units="degrees",
            g1=self.source.e1, g2=self.source.e2,
            w=self.source.w,
            patch_centers=patch_centers,
            npatch=npatch,
        )
        self.config = {
            "min_sep": self.Mpc_h_inv2degree(self.min_rp),
            "max_sep": self.Mpc_h_inv2degree(self.max_rp),
            "nbins": self.n_rp_bins,
            "sep_units": "degree",
            "var_method": var_method,
            "cross_patch_weight": "match",
        }

        ng = NGCorrelation(self.config)
        ng.process(self._lens_Catalog, self._source_Catalog)

        self.dsigma_tangential = ng.xi * self._effective_sigma_crit
        self.dsigma_cross = ng.xi_im * self._effective_sigma_crit
        self.cov = ng.cov * self._effective_sigma_crit**2
        self.mean_rp = self.degree2Mpc_h_inv(ng.meanr)
        self.mean_rp_arcmin = ng.meanr * 180 # degree to arcmin

        if self.randoms is not None:
            self._weighted_npairs = ng.weight
            self._lens_weight_sum = (
                np.sum(self.lens.w) if self.lens.w is not None
                else len(self.lens.ra)
            )
            self._boost_array = np.empty(
                shape=(len(self.randoms), self.n_rp_bins)
            )
            self._dsigma_rand_array = np.empty(
                shape=(2, len(self.randoms), self.n_rp_bins)
            )
            self.boost, self.dsigma_rand = self.calc_rand_corrections()
            self.dsigma_tangential *= self.boost
            self.dsigma_cross *= self.boost
            self.dsigma_tangential -= self.dsigma_rand[0]
            self.dsigma_cross -= self.dsigma_rand[1]
            self.cov *= self.boost**2

        # Multiplicative shear bias correction
        self.dsigma_tangential /= (1 + self.source.m)
        self.dsigma_cross /= (1 + self.source.m)
        self.cov /= (1 + self.source.m)**2


    def calc_rand_corrections(self):
        ng_rand = NGCorrelation(self.config)
        for i in range(len(self.randoms)):
            print(f"Calculating random corrections for random catalogue {i + 1}.")
            random_Catalog = Catalog(
                ra=self.randoms[i].ra, ra_units="degrees",
                dec=self.randoms[i].dec, dec_units="degrees",
                w=self.randoms[i].w,
                patch_centers=self.patch_centers,
                npatch=self.npatch,
            )
            ng_rand.process(random_Catalog, self._source_Catalog)
            random_weight_sum = (
                np.sum(self.randoms[i].w) if self.randoms[i].w is not None
                else len(self.randoms[i].ra)
            )
            self._boost_array[i] = (
                    (self._weighted_npairs / self._lens_weight_sum) /
                    (ng_rand.weight / random_weight_sum)
            )
            self._dsigma_rand_array[0][i] = ng_rand.xi * self._effective_sigma_crit
            self._dsigma_rand_array[1][i] = ng_rand.xi_im * self._effective_sigma_crit
            ng_rand.clear()

        boost = np.average(self._boost_array, axis=0)
        dsigma_rand_tangential = np.average(self._dsigma_rand_array[0], axis=0)
        disgma_rand_cross = np.average(self._dsigma_rand_array[1], axis=0)

        return boost, (dsigma_rand_tangential, disgma_rand_cross)


    def effective_critical_surface_density(self):
        z_l, z_s = np.meshgrid(self.lens.dndz[0], self.source.dndz[0], indexing="ij")
        weights = np.outer(self.lens.dndz[1], self.source.dndz[1])
        sigma_crit_inv = critical_surface_density(z_l, z_s, self.cosmo) ** -1
        sigma_crit_eff = np.average(
            sigma_crit_inv,
            weights=weights,
        ) ** -1
        return sigma_crit_eff


    def save(
            self,
            fname: str,
            save_dir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/",
    ):
        assert fname.endswith(".csv")
        df = pd.DataFrame({
            "mean_rp_Mpc_h_inv": self.mean_rp,
            "mean_rp_arcmin": self.mean_rp_arcmin,
            "npairs": self._weighted_npairs,
            "dsigma_tangential": self.dsigma_tangential,
            "dsigma_cross": self.dsigma_cross,
            "dsigma_stderr": np.sqrt(np.diag(self.cov)),
            "boost": self.boost,
            "dsigma_rand_tangential": self.dsigma_rand[0],
            "dsigma_rand_cross": self.dsigma_rand[1],
        })
        for i in range(len(self.randoms)):
            df[f"boost_randcat_{i + 1:02d}"] = self._boost_array[i]
            df[f"dsigma_rand_tangential_randcat_{i + 1:02d}"] = self._dsigma_rand_array[0][i]
            df[f"dsigma_rand_cross_randcat_{i + 1:02d}"] = self._dsigma_rand_array[1][i]

        df.to_csv(os.path.join(save_dir, fname), index=False)

        # Save covariance matrix
        np.save(
            os.path.join(save_dir, fname.replace(".csv", "_cov.npy")),
            self.cov,
        )

    def degree2Mpc_h_inv(self, degree: np.ndarray | float):
        DM = np.average( # comoving transverse distance in Mpc/h per radian
            a=self.cosmo.comoving_transverse_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        ) * self.cosmo.h
        return np.deg2rad(degree) * DM

    def Mpc_h_inv2degree(self, Mpc_h_inv: np.ndarray | float):
        DM = np.average( # comoving transverse distance in Mpc/h per radian
            a=self.cosmo.comoving_transverse_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        ) * self.cosmo.h
        return np.rad2deg(Mpc_h_inv / DM)



def get_list_Random_catalogues(save_dir=KiDS_RANDOMS_DIR) -> list[Random]:
    """Load all random catalogues from the save_dir."""
    ret = []
    savepaths = natsorted(glob(os.path.join(save_dir, "random_catalogue_*")))
    for savepath in savepaths:
        print("Loading random catalogue from:", savepath)
        ret.append(Random.from_random_catalogue(savepath))
    return ret


def critical_surface_density(
        z_l, z_s, cosmology, comoving=True, d_l=None, d_s=None):
    """Compute the critical surface density.

    Copied from dsigma package:
    https://github.com/johannesulf/dsigma/blob/main/dsigma/physics.py

    Args:
    z_l: Redshift of lens.
    z_s: Redshift of source.
    cosmology: Cosmology to assume for calculations. Only used if comoving
        distances are not passed. If None, use dsigma.default_cosmology.
        Default is None.
    comoving: Flag for using comoving instead of physical units.
        Default is True.
    d_l: Comoving transverse distance to the lens. If not given, it is
        calculated from the redshift provided. Default is None.
    d_s: Comoving transverse distance to the source. If not given, it is
        calculated from the redshift provided. Default is None.
    """
    if d_l is None:
        d_l = cosmology.comoving_transverse_distance(z_l)
    if d_s is None:
        d_s = cosmology.comoving_transverse_distance(z_s)

    with np.errstate(divide="ignore"):
        sigma_crit = c.c**2 / (4 * np.pi * c.G) * (
            (d_s / (1 + z_s)) / (d_l / (1 + z_l)) / ((d_s - d_l) / (1 + z_s)))
    sigma_crit = np.where(d_s <= d_l, np.inf * sigma_crit.unit, sigma_crit)

    if comoving:
        sigma_crit /= (1.0 + z_l)**2

    return sigma_crit.to(
        cu.littleh * u.Msun / u.pc**2, cu.with_H0(cosmology.H0)).value
