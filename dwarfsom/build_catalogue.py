import os
import itertools
import pandas as pd

from astropy.io import fits
from astropy import table

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
_KiDS_photometric_bands = ["u", "g", "r", "i", "Z", "Y", "J", "H", "Ks"]

_KiDS_selected_columns = [
    "ID",
    "KIDS_TILE",
    "THELI_NAME",
    "RAJ2000",
    "DECJ2000",
    "MAG_AUTO",
    "MAGERR_AUTO",
    "EXTINCION_R",
    "FLUX_RADIUS", # TODO: convert unit from pixel to arcsec
    "Z_B",
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

    cat_processed = cat[_KiDS_selected_columns]
    cat_processed["Z_B_ERR"] = (cat["Z_B_MAX"] - cat["Z_B_MIN"]) / 2

    DMAG_R_map = get_DMAG_R_zeropoint_correction()
    cat_processed["DMAG_R"] = [
        DMAG_R_map[tile] for tile in cat_processed["KIDS_TILE"]
    ]

    # Define the mask
    mask = cat["MASK"] & 28668 == 0
    mask *= cat["IMAFLAGS_ISO"] == 0
    mask *= cat["CLASS_STAR"] < 0.5
    mask *= cat["SG2DPHOT"] == 0
    mask *= cat["SG_FLAG"] == 1
    mask *= cat["FLUX_GAAP_r"] / cat["FLUXERR_GAAP_r"] > 5.
    mask *= cat["MAG_AUTO"] < 18.
    mask *= cat["Z_B"] < 1
    for band in _KiDS_photometric_bands:
        mask *= cat[f"FLAG_GAAP_{band}"] == 0

    for band1, band2 in itertools.combinations(_KiDS_photometric_bands, r=2):
        colour = cat[f"MAG_GAAP_{band1}"] - cat[f"MAG_GAAP_{band2}"]
        cat_processed[f"COLOUR_GAAP_{band1}_{band2}"] = colour
        cat_processed[f"COLOURERR_GAAP_{band1}_{band2}"] = (
            cat[f"MAGERR_GAAP_{band1}"]**2 + cat[f"MAGERR_GAAP_{band2}"]**2
        ) ** 0.5
        mask *= (colour < 4) & (colour > -2)

    cat_processed = cat_processed[mask]

    cat_processed.write(
        os.path.join(save_dir, fname),
        format="fits",
        overwrite=True,
    )



def create_GAMA_spectropic_catalogue():
    pass




if __name__ == "__main__":
    if os.uname().nodename.split(".")[0] == "alblas":
        raise EnvironmentError(
            "Do not run on the node alblas. This node has not enough memory."
        )

    get_KiDS_photometric_catalogue()