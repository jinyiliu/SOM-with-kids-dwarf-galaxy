import numpy as np

class KiDS1000:
    _m_bias_dict = {
        1: -0.009,
        2: -0.011,
        3: -0.015,
        4: 0.002,
        5: 0.007,
    }
    _dndz_fname = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/KiDS_DR4/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc"

    @staticmethod
    def get_m_bias(zbin: int) -> float:
        assert zbin in KiDS1000._m_bias_dict
        return KiDS1000._m_bias_dict[zbin]

    @staticmethod
    def get_dndz(zbin: int) -> tuple[np.ndarray, np.ndarray]:
        assert zbin in KiDS1000._m_bias_dict
        fname = KiDS1000._dndz_fname.format(zbin)
        z, dndz = np.loadtxt(fname, unpack=True)
        return z, dndz
