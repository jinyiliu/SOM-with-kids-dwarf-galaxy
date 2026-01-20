import os
import numpy as np
from functools import wraps

from astropy import table
from astropy.io import fits


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


def fits2csv(
        fits_savepath: str,
        csv_savepath: str,
        overwrite: bool=True,
):
    """Convert a FITS catalogue to a CSV catalogue."""
    with fits.open(fits_savepath) as hdul:
         cat = table.Table(hdul[1].data)

    cat.write(csv_savepath, format="csv", overwrite=overwrite)

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