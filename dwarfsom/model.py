import numpy as np
import pyccl as ccl
from onepower import Spectra

from ._pk2real import PkTransformer

def Pgm2DSigma(
        model: Spectra,
        rpmin=None,
        rpmax=None,
        components: bool=False,
):
    transformer = PkTransformer(
        corr_type="ds",
        model=model,
        k_min=None,
        k_max=None,
        sep_min_in=rpmin,
        sep_max_in=rpmax,
        n_transform=None,
        z_s=None,
        components=components,
    )
    return transformer()