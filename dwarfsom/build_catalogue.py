import os
import itertools
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy import table

def Rhf2FWHM(Rhf):
    """Convert half-light radius to FWHM for a Gaussian profile."""
    return Rhf * 1.75

def create_mask_for_candidate_dwarfs(
    magr, gminusr, gminusr_err, mueff, mueff_err,
):
    mask = gminusr - gminusr_err < 0.00085 * (magr - 13.) ** 3 + 0.83
    mask *= mueff + mueff_err > 16.7 + 0.7 * (magr - 13.)
    return mask


_NODE_DIR = "/net/alblas"
_DATA_DIR = os.path.join(
    _NODE_DIR,
    "data1/jliu/SOM-with-kids-dwarf-galaxy/data",
)
_KiDS_DIR = os.path.join(
    _DATA_DIR,
    "KiDS_DR4",
)
_KiDS_RAW_DATA_DIR = os.path.join(
    _KiDS_DIR,
    "ugriZYJHKs_tile_cats",
)
_KiDS_DMAG_path = os.path.join(
    _KiDS_DIR,
    "KiDS_DMAG_R_zeropoint_corrections.csv",
)

_KiDS_OmegaCAM_pixel_length = 0.213 # arcsec
KiDS_photometric_bands = ["u", "g", "r", "i", "Z", "Y", "J", "H", "Ks"]

_KiDS_selected_columns = [
    "ID",
    "KIDS_TILE",
    "THELI_NAME",
    "RAJ2000",
    "DECJ2000",
    "MAG_AUTO",
    "MAGERR_AUTO",
    "DMAG_R",
    "EXTINCTION_r",
    "FLUX_RADIUS",
    "FWHM_IMAGE",
    "Z_B",
    "MU_MAX", # Peak surface brightness above background
]

_GAMA_DIR = os.path.join(
    _DATA_DIR,
    "GAMA_DR4",
)
_GAMA_gkvScienceCat_path = os.path.join(
    _GAMA_DIR,
    "gkvScienceCatv02.fits",
)
_GAMA_StellarMasses_path = os.path.join(
    _GAMA_DIR,
    "StellarMassesGKVv24.fits",
)
_GAMA_selected_columns_gkvScienceCat = [
    "uberID",
    "RAcen", # RA of flux-weighted centre (ICRS)
    "Deccen", # Dec of flux-weighted centre (ICRS)
    "uberclass",
    "mag",
    "Z",
    "NQ",
    "SC",
    "R50", # Approximate elliptical semi-major axis containing 50% of the flux
]
_GAMA_selected_columns_StellarMasses = [
    "uberID",
    "nefffilt",
    "nefftemp",
    "mstar",
    "delmstar",
    "logmstar",
    "dellogmstar",
    "ppp",
]


def get_DMAG_R_zeropoint_correction() -> pd.DataFrame:
    if not os.path.exists(_KiDS_DMAG_path):
        t = {
            "KIDS_TILE": [],
            "DMAG_R": [],
        }
        for tile_catalogue in os.listdir(_KiDS_RAW_DATA_DIR):
            t["KIDS_TILE"].append(
                "KIDS_" + "_".join(tile_catalogue.split("_")[2:4])
            )
            with fits.open(os.path.join(_KiDS_RAW_DATA_DIR, tile_catalogue)) as hdul:
                t["DMAG_R"].append(
                    hdul[0].header["DMAG_R"]
                )
        df = pd.DataFrame(t)
        df.to_csv(_KiDS_DMAG_path, index=False)

    df = pd.read_csv(_KiDS_DMAG_path, index_col="KIDS_TILE")
    return df["DMAG_R"].to_dict()


def create_KiDS_photometric_catalogue(
        save_dir: str=_KiDS_DIR,
        fname: str="KiDS_panchromatic_catalogue.fits",
):
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist.")

    merged_photometry_cat_path = os.path.join(
        "/net/eemmeer",
        "data2/KiDS/KiDS-1000/ESO-DR4-photometry-catalogues",
        "KiDS.DR4.merged.fits",
    )

    with fits.open(merged_photometry_cat_path) as hdul:
        cat = table.Table(hdul[1].data)

    DMAG_R_map = get_DMAG_R_zeropoint_correction()
    cat["DMAG_R"] = [
        DMAG_R_map[tile.strip()] for tile in cat["KIDS_TILE"]
    ]

    cat_processed = cat[_KiDS_selected_columns]
    cat_processed["Z_B_ERR"] = (cat["Z_B_MAX"] - cat["Z_B_MIN"]) / 2

    # Define the mask
    mask = cat["MASK"] & 28668 == 0
    mask *= cat["IMAFLAGS_ISO"] == 0
    mask *= cat["CLASS_STAR"] < 0.5
    mask *= cat["SG2DPHOT"] == 0
    mask *= cat["SG_FLAG"] == 1
    mask *= cat["MAG_AUTO"] + cat["DMAG_R"] < 19.65

    for band in KiDS_photometric_bands:
        mask *= cat[f"FLAG_GAAP_{band}"] == 0

    for band1, band2 in itertools.combinations(KiDS_photometric_bands, r=2):
        colour = cat[f"MAG_GAAP_{band1}"] - cat[f"MAG_GAAP_{band2}"]
        cat_processed[f"COLOUR_GAAP_{band1}_{band2}"] = colour
        cat_processed[f"COLOURERR_GAAP_{band1}_{band2}"] = (
            cat[f"MAGERR_GAAP_{band1}"]**2 + cat[f"MAGERR_GAAP_{band2}"]**2
        ) ** 0.5
        mask *= cat_processed[f"COLOURERR_GAAP_{band1}_{band2}"] < 0.2

    cat_processed = cat_processed[mask]

    # Additional derived columns
    cat_processed["FLUX_RADIUS"] = cat_processed["FLUX_RADIUS"] * _KiDS_OmegaCAM_pixel_length
    cat_processed["FWHM_IMAGE"] = cat_processed["FWHM_IMAGE"] * _KiDS_OmegaCAM_pixel_length
    cat_processed["MAG_CORR"] = (
        cat_processed["MAG_AUTO"] + cat_processed["DMAG_R"] - cat_processed["EXTINCTION_r"]
    )
    cat_processed["MU_EFF_FLUX_RADIUS"] = (
        cat_processed["MAG_AUTO"] + 2.5 * np.log10(
            2 * np.pi * (Rhf2FWHM(cat_processed["FLUX_RADIUS"]) / 2) ** 2
        )
    )
    cat_processed["MU_EFF_FWHM_IMAGE"] = (
        cat_processed["MAG_AUTO"] + 2.5 * np.log10(
            2 * np.pi * (cat_processed["FWHM_IMAGE"] / 2) ** 2
        )
    )

    cat_processed.write(
        os.path.join(save_dir, fname),
        format="fits",
        overwrite=True,
    )



def create_GAMA_spectroscopic_catalogue(
        save_dir: str=_GAMA_DIR,
        fname: str="GAMA_processed_catalogue.fits",
):
    with fits.open(_GAMA_gkvScienceCat_path) as hdul:
         dmu_gkvScienceCat = table.Table(hdul[1].data)

    with fits.open(_GAMA_StellarMasses_path) as hdul:
         dmu_StellarMassesGKV = table.Table(hdul[1].data)

    cat = table.join(
        dmu_gkvScienceCat[_GAMA_selected_columns_gkvScienceCat],
        dmu_StellarMassesGKV[_GAMA_selected_columns_StellarMasses],
        keys="uberID",
    )

    del dmu_gkvScienceCat, dmu_StellarMassesGKV

    cat.write(
        os.path.join(save_dir, "GAMA_joined_catalogue.fits"),
        format="fits",
        overwrite=True,
    )

    mask = cat["SC"] > 3

    cat_processed = cat[mask]
    cat_processed.write(
        os.path.join(save_dir, fname),
        format="fits",
        overwrite=True,
    )


def fits2csv(
        fits_savepath: str,
        csv_savepath: str,
        overwrite: bool=True,
):
    """Convert a FITS catalogue to a CSV catalogue."""
    with fits.open(fits_savepath) as hdul:
         cat = table.Table(hdul[1].data)

    cat.write(csv_savepath, format="csv", overwrite=overwrite)


if __name__ == "__main__":
    if os.uname().nodename.split(".")[0] == "alblas":
        raise EnvironmentError(
            "Do not run on the node alblas. This node has not enough memory."
        )

    get_KiDS_photometric_catalogue()