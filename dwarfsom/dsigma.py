import os.path
from glob import glob
import numpy as np
import pandas as pd

from natsort import natsorted
from dataclasses import dataclass
from astropy.cosmology import FlatLambdaCDM
from treecorr import NGCorrelation, Catalog
from dsigma.physics import critical_surface_density
from dwarfsom.utils import calculate_sky_area

cosmo_default = FlatLambdaCDM(H0=100, Om0=0.3)

@dataclass
class Lens:
    ra: np.ndarray
    dec: np.ndarray
    dndz: tuple[np.ndarray, np.ndarray]
    dndmstar: tuple[np.ndarray, np.ndarray]
    w: np.ndarray | None = None

    def __eq__(self, other):
        if not isinstance(other, Lens):
            return False
        ret = (
            np.array_equal(self.ra, other.ra) and
            np.array_equal(self.dec, other.dec) and
            np.array_equal(self.dndz[0], other.dndz[0]) and
            np.array_equal(self.dndz[1], other.dndz[1]) and
            np.array_equal(self.dndmstar[0], other.dndmstar[0]) and
            np.array_equal(self.dndmstar[1], other.dndmstar[1]) and
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

    @staticmethod
    def build_random_catalogues(
            n_randoms: int,
            n_catalogues: int,
            savedir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/randoms/",
    ):
        RA_Dec_ranges = [
            [(329.5, 360.), (-36.6, -25.7)], # KiDS-S
            [(0., 53.5), (-36.6, -25.7)],  # KiDS-S
            [(156., 238.), (-5., 4.)], # KiDS-N
            [(128.5, 141.7), (-2., 3.)], # KiDS-N-W2
        ]
        sky_areas = [
            calculate_sky_area(RA_range, Dec_range)
            for RA_range, Dec_range in RA_Dec_ranges
        ]
        n_randoms_field = [
            int(np.round(n_randoms * sky_area / sum(sky_areas)))
            for sky_area in sky_areas
        ]

        fname = "random_catalogue_{:02d}.csv"
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(n_catalogues):
            coords = np.empty((2, n_randoms))
            for field in range(len(RA_Dec_ranges)):
                RA_range, Dec_range = RA_Dec_ranges[field]
                ra_random = np.random.uniform(
                    low=RA_range[0],
                    high=RA_range[1],
                    size=n_randoms_field[field],
                )
                dec_random = np.degrees(np.arcsin(
                    np.random.uniform(
                        low=np.sin(np.radians(Dec_range[0])),
                        high=np.sin(np.radians(Dec_range[1])),
                        size=n_randoms_field[field],
                    )
                ))
                start_idx = sum(n_randoms_field[:field])
                end_idx = start_idx + n_randoms_field[field]
                coords[0, start_idx:end_idx] = ra_random
                coords[1, start_idx:end_idx] = dec_random

            df = pd.DataFrame({
                "RAJ2000": coords[0],
                "DECJ2000": coords[1],
            })
            savepath = os.path.join(savedir, fname.format(i + 1))
            df.to_csv(savepath, index=False)
            print(f"Saved random catalogue {i + 1} to {savepath}.")


    @classmethod
    def from_random_catalogue(cls, savepath: str) -> "Random":
        df = pd.read_csv(savepath)
        ra = df["RAJ2000"].values
        dec = df["DECJ2000"].values
        return cls(ra=ra, dec=dec)


@dataclass
class DSigmaData:
    mean_rp: np.ndarray
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
    def from_file(
            cls,
            fname: str,
            savedir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/",
    ) -> "DSigmaData":
        df = pd.read_csv(os.path.join(savedir, fname))
        boost_array = df.filter(like="boost_randcat").values.T
        dsigma_rand_array = np.vstack((
            [df.filter(like="dsigma_rand_tangential_randcat").values.T],
            [df.filter(like="dsigma_rand_cross_randcat").values.T],
        ))
        return cls(
            mean_rp=df["mean_rp_hMpc"].values,
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
            cosmo=cosmo_default,
            n_rp_bins: int=8,
            min_rp: float=0.01,
            max_rp: float=10.0,
    ):
        """Excess Surface Density Estimator using TreeCorr."""
        self.lens = lens
        self.source = source
        self.randoms = randoms
        self.cosmo = cosmo
        self.n_rp_bins = n_rp_bins
        self.min_rp = min_rp
        self.max_rp = max_rp

        self._effective_sigma_crit = self.effective_critical_surface_density()

        self.lens_Catalog = Catalog(
            ra=lens.ra, ra_units="degrees",
            dec=lens.dec, dec_units="degrees",
            w=self.lens.w,
        )
        self.source_Catalog = Catalog(
            ra=self.source.ra, ra_units="degrees",
            dec=self.source.dec, dec_units="degrees",
            g1=self.source.e1, g2=self.source.e2,
            w=self.source.w,
        )
        self.config = {
            "min_sep": self.hMpc2degree(self.min_rp),
            "max_sep": self.hMpc2degree(self.max_rp),
            "nbins": self.n_rp_bins,
            "sep_units": "degree",
        }

        ng = NGCorrelation(self.config)
        ng.process(self.lens_Catalog, self.source_Catalog)

        self.dsigma_tangential = ng.xi * self._effective_sigma_crit
        self.dsigma_cross = ng.xi_im * self._effective_sigma_crit
        self.cov = ng.cov * self._effective_sigma_crit**2
        self.mean_rp = self.degree2hMpc(ng.meanr)

        if self.randoms is not None:
            self._weighted_npairs = ng.weight
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
            )
            ng_rand.process(random_Catalog, self.source_Catalog)
            self._boost_array[i] = (
                    (self._weighted_npairs / len(self.lens.ra)) /
                    (ng_rand.weight / len(self.randoms[i].ra))
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
        sigma_crit = critical_surface_density(z_l, z_s, self.cosmo)
        finite_mask = np.isfinite(sigma_crit)
        sigma_crit_eff = np.average(
            sigma_crit[finite_mask],
            weights=weights[finite_mask],
        )
        return sigma_crit_eff


    def save(
            self,
            fname: str,
            savedir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/",
    ):
        assert fname.endswith(".csv")
        df = pd.DataFrame({
            "mean_rp_hMpc": self.mean_rp,
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

        df.to_csv(os.path.join(savedir, fname), index=False)


    def degree2hMpc(self, degree: np.ndarray | float):
        DA = np.average(
            a=self.cosmo.angular_diameter_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        )
        radian = np.radians(degree)
        hMpc = radian * DA
        return hMpc


    def hMpc2degree(self, hMpc: np.ndarray | float):
        DA = np.average(
            a=self.cosmo.angular_diameter_distance(self.lens.dndz[0]).value,
            weights=self.lens.dndz[1],
        )
        radian = hMpc / DA
        degree = np.degrees(radian)
        return degree


def get_combined_dsigma(dsigma_list: list[DSigma]):
    """Combine multiple DSigma measurements by inverse-variance weighting."""
    assert dsigma_list[0].lens == dsigma_list[1].lens == dsigma_list[2].lens
    mean_rp = dsigma_list[0].mean_rp
    dsigma_tangential = np.tile(mean_rp, (len(dsigma_list), 1))
    weights = np.tile(mean_rp, (len(dsigma_list), 1))
    var = np.tile(mean_rp, (len(dsigma_list), 1))

    for i, dsigma in enumerate(dsigma_list):
        dsigma_tangential[i] = dsigma.dsigma_tangential
        weights[i] = np.diag(dsigma.cov) ** -1
        var[i] = np.diag(dsigma.cov)

    dsigma_tangential = np.average(
        dsigma_tangential,
        weights=weights,
        axis=0,
    )
    var = 1 / np.sum(1 / var, axis=0)

    return mean_rp, dsigma_tangential, var


def get_list_Random_catalogues(
        savedir="/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/randoms/",
) -> list[Random]:
    """Load all random catalogues from the savedir."""
    ret = []
    savepaths = natsorted(glob(os.path.join(savedir, "*.csv")))
    for savepath in savepaths:
        print("Loading random catalogue from:", savepath)
        ret.append(Random.from_random_catalogue(savepath))
    return ret


