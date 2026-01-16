import os
import numpy as np
import pandas as pd

class KiDSWL:
    _m_bias_dict = {
        1: -0.009,
        2: -0.011,
        3: -0.015,
        4: 0.002,
        5: 0.007,
    }
    _dndz_fname = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/KiDS_DR4/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"
    _gold_WL_cat_path = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/KiDS_DR4/KiDS_DR4.1_gold_WL_cat/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.csv"
    gold = pd.read_csv(_gold_WL_cat_path)

    @staticmethod
    def get_m_bias(zbin: int) -> float:
        assert zbin in KiDSWL._m_bias_dict
        return KiDSWL._m_bias_dict[zbin]

    @staticmethod
    def get_dndz(zbin: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized dN/dz for the given redshift bin."""
        assert zbin in KiDSWL._m_bias_dict
        fname = KiDSWL._dndz_fname.format(zbin)
        z, dndz = np.loadtxt(fname, unpack=True)
        return z, dndz


class KiDSDwarf:
    _cat_path = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/SOM/kids_dwarfs.csv"
    _dndz_path = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/SOM/z_pdfs_by_stellar_mass_bins.csv"
    _dndmstar_path = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/SOM/logmstar_pdfs_by_stellar_mass_bins.csv"
    _dndz_df = pd.read_csv(_dndz_path)
    _dndmstar_df = pd.read_csv(_dndmstar_path)

    dwarf = pd.read_csv(_cat_path)

    @staticmethod
    def get_dndz(mbin: int, bin_center: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized dN/dz for the given stellar mass bin."""
        z = KiDSDwarf._dndz_df["BIN_START"].values
        if bin_center:
            z += KiDSDwarf._dndz_df["BIN_END"].values
            z /= 2
        dndz = KiDSDwarf._dndz_df[f"BIN_{mbin}"].values
        return z, dndz

    @staticmethod
    def get_dndmstar(mstarbin: int, bin_center: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized dN/dlogM* for the given stellar mass bin."""
        logmstar = KiDSDwarf._dndmstar_df["BIN_START"].values
        if bin_center:
            logmstar += KiDSDwarf._dndmstar_df["BIN_END"].values
            logmstar /= 2
        dndmstar = KiDSDwarf._dndmstar_df[f"BIN_{mstarbin}"].values
        return logmstar, dndmstar
