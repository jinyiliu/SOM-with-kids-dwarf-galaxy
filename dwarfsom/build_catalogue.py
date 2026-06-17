import os
import itertools
import warnings
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy import table
from functools import wraps

warnings.filterwarnings(action="ignore", category=RuntimeWarning)

def prevent_on_server(server_name: str="alblas"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.uname().nodename.split(".")[0] == server_name:
                raise EnvironmentError(
                    f"Do not run on the node {server_name}. "
                    f"This node has not enough memory."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def Rhf2FWHM(Rhf):
    """Convert half-light radius to FWHM for a Gaussian profile."""
    return Rhf * 1.75


_NODE_DIR = "/net/alblas"
_DATA_DIR = os.path.join(_NODE_DIR, "data1/jliu/SOM-with-kids-dwarf-galaxy/data")

_KiDS_DIR = os.path.join(_DATA_DIR, "KiDS_DR4")
_KiDS_TILE_DATA_DIR = os.path.join(_KiDS_DIR, "ugriZYJHKs_tile_cats")
_KiDS_DMAG_path = os.path.join(_KiDS_DIR, "KiDS_DMAG_R_zeropoint_corrections.csv")
_KiDS_WL_DATA_DIR = os.path.join(_KiDS_DIR, "KiDS_DR4.1_gold_WL_cat")
_KiDS_BGS_DIR = os.path.join(_KiDS_DIR, "BrightGalaxySample")
_KiDS_gold_WL_FITS_path = os.path.join(_KiDS_WL_DATA_DIR, "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits")
KiDS_gold_WL_CSV_path = os.path.join(_KiDS_WL_DATA_DIR, "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.csv")
_KiDS_BASIC_RANDOMS_DIR = os.path.join(_KiDS_DIR, "BASIC_RANDOMS")
KiDS_RANDOMS_DIR = os.path.join(_KiDS_DIR, "randoms")

_KiDS_OmegaCAM_pixel_length = 0.213 # arcsec
KiDS_photometric_bands = ["u", "g", "r", "i", "Z", "Y", "J", "H", "Ks"]

_KiDS_EXCLUDED_tiles = [ # tiles that are in BASIC RANDOMS but not in KiDS DR4 footprint
    "28p5_m29p2",
    "48p2_m27p2",
    "195p0_1p5",
    "210p0_m0p5",
    "356p5_m30p2",
]

_KiDS_selected_columns = [
    "ID",
    "KIDS_TILE",
    "THELI_NAME",
    "RAJ2000",
    "DECJ2000",
    "MAG_AUTO",
    "MAG_GAAP_r",
    "MAGERR_AUTO",
    "DMAG_R",
    "EXTINCTION_r",
    "FLUX_RADIUS",
    "FWHM_IMAGE",
    "Z_B",
    "MU_MAX", # Peak surface brightness above background
]

_KiDS_gold_WL_selected_columns = [
    "RAJ2000",
    "DECJ2000",
    "e1",
    "e2",
    "weight",
    "Z_B",
]

_KiDS_BGS_selected_columns = [
    "ID",
    "RAJ2000",
    "DECJ2000",
    "REDSHIFT",
    "MASS_MED",
    "MASS_BEST",
    "MAG_GAAP_r",
    "MAG_AUTO",
]

_GAMA_DIR = os.path.join(_DATA_DIR, "GAMA_DR4")
_GAMA_gkvScienceCat_path = os.path.join(_GAMA_DIR, "gkvScienceCatv02.fits")
_GAMA_StellarMasses_path = os.path.join(_GAMA_DIR, "StellarMassesGKVv24.fits")

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
        for tile_catalogue in os.listdir(_KiDS_TILE_DATA_DIR):
            t["KIDS_TILE"].append(
                "KIDS_" + "_".join(tile_catalogue.split("_")[2:4])
            )
            with fits.open(os.path.join(_KiDS_TILE_DATA_DIR, tile_catalogue)) as hdul:
                t["DMAG_R"].append(
                    hdul[0].header["DMAG_R"]
                )
        df = pd.DataFrame(t)
        df.to_csv(_KiDS_DMAG_path, index=False)

    df = pd.read_csv(_KiDS_DMAG_path, index_col="KIDS_TILE")
    return df["DMAG_R"].to_dict()


@prevent_on_server("alblas")
def build_KiDS_preselected_candidates(
        save_dir: str=_KiDS_DIR,
        fname: str="KiDS_preselected_candidates.fits",
        overwrite: bool=False,
):
    """Create a masked KiDS panchromatic catalogue with selected columns and
    derived columns that are relevant for dwarf galaxy candidates preselection.
    """
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
    cat_processed["FLUX_GAAP_SNR_r"] = cat["FLUX_GAAP_r"] / cat["FLUXERR_GAAP_r"]
    cat_processed["MAG_CORR"] = (
        cat_processed["MAG_AUTO"] + cat_processed["DMAG_R"] - cat_processed["EXTINCTION_r"]
    )

    # Define the mask
    mask = cat["MASK"] & 28668 == 0
    mask *= cat["IMAFLAGS_ISO"] == 0
    mask *= cat["CLASS_STAR"] < 0.5
    mask *= cat["SG2DPHOT"] == 0
    mask *= cat["SG_FLAG"] == 1
    for band in KiDS_photometric_bands:
        mask *= cat[f"FLAG_GAAP_{band}"] == 0

    mask *= cat["MAG_AUTO"] + cat["DMAG_R"] < 20.5
    mask *= cat["Z_B"] < 1.0
    mask *= cat_processed["FLUX_GAAP_SNR_r"] > 5

    mask *= cat["FWHM_IMAGE"] > 0.
    mask *= cat["FLUX_RADIUS"] > 0.
    for band in KiDS_photometric_bands:
        mask *= cat[f"FLUX_GAAP_{band}"] > 0
        mask *= cat[f"FLUXERR_GAAP_{band}"] > 0

    for band1, band2 in itertools.combinations(KiDS_photometric_bands, r=2):
        colour = cat[f"MAG_GAAP_{band1}"] - cat[f"MAG_GAAP_{band2}"]
        cat_processed[f"COLOUR_GAAP_{band1}_{band2}"] = colour
        cat_processed[f"COLOURERR_GAAP_{band1}_{band2}"] = (
            cat[f"MAGERR_GAAP_{band1}"]**2 + cat[f"MAGERR_GAAP_{band2}"]**2
        ) ** 0.5

    # Additional derived columns
    cat_processed["FLUX_RADIUS"] = cat_processed["FLUX_RADIUS"] * _KiDS_OmegaCAM_pixel_length
    cat_processed["FWHM_IMAGE"] = cat_processed["FWHM_IMAGE"] * _KiDS_OmegaCAM_pixel_length
    cat_processed["MU_EFF_FLUX_RADIUS"] = (
        cat_processed["MAG_CORR"] + 2.5 * np.log10(
            2 * np.pi * (Rhf2FWHM(cat_processed["FLUX_RADIUS"]) / 2) ** 2
        )
    )
    cat_processed["MU_EFF_FWHM_IMAGE"] = (
        cat_processed["MAG_CORR"] + 2.5 * np.log10(
            2 * np.pi * (cat_processed["FWHM_IMAGE"] / 2) ** 2
        )
    )

    cat_preselected = cat_processed[mask]
    cat_preselected.write(
        os.path.join(save_dir, fname),
        format="fits",
        overwrite=overwrite,
    )


@prevent_on_server("alblas")
def build_KiDS_random_catalogues(
        fraction_per_tile: int=0.06,
        save_dir: str=KiDS_RANDOMS_DIR,
        overwrite: bool=False,
):
    """Build KiDS random catalogues by sampling from the KiDS BASIC RANDOMS.

    Args:
        fraction_per_tile: Fraction of the randoms per tile to include in each catalogue.
        save_dir:
        overwrite:
    """
    assert 0. < fraction_per_tile < 1.

    tiles = [
        tile for tile in os.listdir(_KiDS_BASIC_RANDOMS_DIR)
        if not any(excluded in tile for excluded in _KiDS_EXCLUDED_tiles)
    ]

    for seqnr in range(1, 10):
        fname = f"random_catalogue_{seqnr:02d}.csv"
        tables = []
        for tile in tiles:
            print("Processing BASIC RANDOM tile catalogue:", tile)
            with fits.open(os.path.join(_KiDS_BASIC_RANDOMS_DIR, tile)) as hdul:
                t = table.Table(hdul[1].data)

            t = t[:round(len(t) * fraction_per_tile)]
            mask = t["MASK"] & 28668 == 0
            mask *= t["SeqNr"] == seqnr
            t = t[mask]
            tables.append(t[["ALPHA_J2000", "DELTA_J2000"]])

        merged_table = table.vstack(tables, join_type="exact")
        merged_table.write(
            os.path.join(save_dir, fname),
            format="csv",
            overwrite=overwrite,
        )


def build_GAMA_spectroscopic_catalogue(
        save_dir: str=_GAMA_DIR,
        fname: str="GAMA_joined_catalogue_SCge4.fits",
        overwrite: bool=False,
):
    """Build a GAMA spectroscopic catalogue combining gkvScienceCat and
    StellarMassesGKV, with selected columns and masked by SC > 3. The output
    catalogue is saved in FITS format for subsequent processing in TOPCAT
    sofware.
    """
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
        overwrite=overwrite,
    )

    mask = cat["SC"] > 3
    mask *= cat["uberclass"] == 1
    mask *= cat["NQ"] > 2

    cat_processed = cat[mask]
    cat_processed.write(
        os.path.join(save_dir, fname),
        format="fits",
        overwrite=overwrite,
    )


@prevent_on_server("alblas")
def save_KiDS_gold_WL_csv_cat_with_selected_columns(
        save_dir: str=_KiDS_WL_DATA_DIR,
        fname: str=os.path.basename(KiDS_gold_WL_CSV_path),
        overwrite: bool=False,
):
    """Save a CSV version of the KiDS gold WL catalogue with selected columns."""
    with fits.open(_KiDS_gold_WL_FITS_path) as hdul:
        cat = table.Table(hdul[1].data)

    cat = cat[_KiDS_gold_WL_selected_columns]
    cat["TOMO_BIN"] = np.zeros(len(cat), dtype=int)
    cat["TOMO_BIN"][(cat["Z_B"] > 0.1) & (cat["Z_B"] <= 0.3)] = 1
    cat["TOMO_BIN"][(cat["Z_B"] > 0.3) & (cat["Z_B"] <= 0.5)] = 2
    cat["TOMO_BIN"][(cat["Z_B"] > 0.5) & (cat["Z_B"] <= 0.7)] = 3
    cat["TOMO_BIN"][(cat["Z_B"] > 0.7) & (cat["Z_B"] <= 0.9)] = 4
    cat["TOMO_BIN"][(cat["Z_B"] > 0.9) & (cat["Z_B"] <= 1.2)] = 5

    cat.write(
        os.path.join(save_dir, fname), format="csv", overwrite=overwrite)


def build_KiDS_BrightGalaxySample_catalogue(
        savedir: str=_KiDS_BGS_DIR,
        fname: str="processed_KiDS_DR4_brightsample.csv",
):
    kids_preselected_path = os.path.join(
        _KiDS_DIR, "KiDS_preselected_candidates.csv"
    )
    if not os.path.exists(kids_preselected_path):
        raise ValueError(
            "Please generate the \"KiDS_preselected_candidates.csv\" first."
        )
    kids_preselected = pd.read_csv(
        kids_preselected_path,
    )
    kids_bgs = pd.read_csv(
        os.path.join(
            _KiDS_BGS_DIR,
            "KiDS_DR4_brightsample_LePhare.csv",
        ),
    )
    combined = pd.merge(
        kids_bgs,
        kids_preselected,
        on="ID",
        how="inner",
    )
    combined["RAJ2000"] = combined["RAJ2000_x"]
    combined["DECJ2000"] = combined["DECJ2000_y"]
    combined = combined[_KiDS_BGS_selected_columns]
    combined["MASS_BEST_CALIB"] = combined["MASS_BEST"] + (
        (combined["MAG_GAAP_r"] - combined["MAG_AUTO"]) / 2.5
    )
    combined.to_csv(
        os.path.join(savedir, fname)
    )



if __name__ == "__main__":
    build_KiDS_random_catalogues()
    build_KiDS_preselected_candidates()
    build_GAMA_spectroscopic_catalogue()
    save_KiDS_gold_WL_csv_cat_with_selected_columns()
    build_KiDS_BrightGalaxySample_catalogue()
