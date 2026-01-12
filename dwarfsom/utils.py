import os
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

