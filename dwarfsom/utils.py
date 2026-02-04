import os
import numpy as np
import pandas as pd

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
    from dwarfsom.build_catalogue import _KiDS_BASIC_RANDOMS_DIR
    fnames = os.listdir(_KiDS_BASIC_RANDOMS_DIR)
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


def build_KiDS_random_catalogues(
        n_randoms: int,
        n_catalogues: int,
        savedir: str,
):
    """Build random catalogues for KiDS DR4 given the RA and Dec ranges of the
    KiDS fields. The random catalogues are saved as FITS files in the specified
    directory.

    DEPRECATED: Use the official BASIC_RANDOMS catalogues instead.
    """
    RA_Dec_ranges = [
        [(330., 360.), (-35.6, -27.)], # KiDS-S
        [(0., 53.9), (-35.6, -27.)],  # KiDS-S
        [(157., 237.3), (-4., 3.)], # KiDS-N
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

    fname = "random_catalogue_{:02d}.fits"
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
        t = table.Table.from_pandas(df)
        savepath = os.path.join(savedir, fname.format(i + 1))
        t.write(savepath, format="fits", overwrite=True)
        print(f"Saved random catalogue {i + 1} to {savepath}.")