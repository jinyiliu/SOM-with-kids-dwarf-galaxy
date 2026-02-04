import os
import numpy as np
from astropy import table
from astropy.io import fits

def fits2csv(
        fits_savepath: str,
        csv_savepath: str,
        overwrite: bool=True,
):
    """Convert a FITS catalogue to a CSV catalogue."""
    with fits.open(fits_savepath) as hdul:
         cat = table.Table(hdul[1].data)

    cat.write(csv_savepath, format="csv", overwrite=overwrite)

def csv2fits(
        csv_savepath: str,
        fits_savepath: str,
        overwrite: bool=True,
):
    """Convert a CSV catalogue to a FITS catalogue."""
    cat = table.Table.read(csv_savepath, format="csv")
    cat.write(fits_savepath, format="fits", overwrite=overwrite)


def calculate_sky_area(ra_range: tuple[float], dec_range: tuple[float]):
    """
    Calculate the sky area in square degrees given RA and Dec ranges.
    RA is in degrees with 0 <= RA < 360.
    Dec is in degrees with -90 <= Dec <= 90.
    """
    DeltaRA = ra_range[1] - ra_range[0]
    if DeltaRA < 0:
        DeltaRA += 360.
    DeltaSinDec = np.sin(np.radians(dec_range[1])) - np.sin(np.radians(dec_range[0]))
    area = DeltaRA * np.degrees(DeltaSinDec)
    return area


# The tile sets are different for BASIC RANDOMS and KiDS DR4.
# KiDS DR4 has 1006 tiles, while BASIC RANDOMS has 1011 tiles.
def get_BASIC_RANDOMS_tile_set() -> set[tuple[float, float]]:
    """Get a set of BASIC_RANDOMS tile (ra, dec) centers."""
    from dwarfsom.build_catalogue import _KiDS_RANDOMS_DIR
    fnames = os.listdir(_KiDS_RANDOMS_DIR)
    tiles = {
        (
            float(fname.split("_")[1].replace("p", ".")),
            float(fname.split("_")[2].replace("p", ".").replace("m", "-"))
         )
        for fname in fnames
    }
    return tiles


def get_KiDS_DR4_tile_set() -> set[tuple[float, float]]:
    """Get a set of KiDS DR4 tile (ra, dec) centers."""
    from dwarfsom.build_catalogue import _KiDS_TILE_DATA_DIR
    fnames = os.listdir(_KiDS_TILE_DATA_DIR)
    tiles = {
        (float(fname.split("_")[2]), float(fname.split("_")[3]))
        for fname in fnames
    }
    return tiles