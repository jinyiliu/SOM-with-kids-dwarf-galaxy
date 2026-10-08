import os
import gc
import numpy as np
import pandas as pd

from dwarfsom.dsigma import DSigma, Lens, Source, Random


if __name__ == "__main__":
    _SOM_data = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/SOM/"
    _KiDS_data = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/KiDS_DR4/"
    save_dir = "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/"

    # region Construct lens1 and lens2
    dwarfs = pd.read_csv(_SOM_data + "kids_dwarfs.csv")
    dndz = pd.read_csv(_SOM_data + "dndz_by_mass_bins.csv")
    dndlogmstar = pd.read_csv(_SOM_data + "dndlogmstar_by_mass_bins.csv")

    lens1, lens2 = [
        Lens(
            ra=dwarfs[dwarfs["MASS_BIN"] == mass_bin]["RAJ2000"],
            dec=dwarfs[dwarfs["MASS_BIN"] == mass_bin]["DECJ2000"],
            dndz=(
                dndz["BIN_CENTRE"],
                dndz[f"MASS_BIN_{mass_bin}"],
            ),
            dndlogmstar=(
                dndlogmstar["BIN_CENTRE"],
                dndlogmstar[f"MASS_BIN_{mass_bin}"],
            ),
            w=dwarfs[dwarfs["MASS_BIN"] == mass_bin]["weight"],
        )
        for mass_bin in [1, 2]
    ]
    del dwarfs, dndz, dndlogmstar
    # endregion

    # region Construct source with KiDS-1000 gold WL catalogue
    # m_bias = {
    #     1: -0.013,
    #     2: -0.010,
    #     3: -0.011,
    #     4: 0.007,
    #     5: 0.006,
    # }
    #
    # gold_WL = pd.read_csv(os.path.join(
    #     _KiDS_data,
    #     "KiDS_DR4.1_gold_WL_cat",
    #     "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.csv",
    # ))
    #
    # n_excluded_tomo_bins = 1
    #
    # tomo_weights = {}
    # for tomo in range(1, 6):
    #     tomo_weights[tomo] = float(
    #         gold_WL[gold_WL["TOMO_BIN"] == tomo]["weight"].sum()
    #     )
    #
    # m = np.average(  # compute average m-bias
    #     list(m_bias.values())[n_excluded_tomo_bins:],
    #     weights=list(tomo_weights.values())[n_excluded_tomo_bins:],
    # )
    #
    # dndz = []
    # for tomo in range(1, 6):
    #     som_dndz = np.loadtxt(
    #         os.path.join(
    #             _KiDS_data,
    #             "SOM_N_of_Z",
    #             f"K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_SOMcols_Fid_blindC_TOMO{tomo:d}_Nz.asc",
    #         ),
    #         skiprows=1,
    #         unpack=True,
    #     )
    #
    # dndz = (
    #     som_dndz[0],
    #     np.average(
    #         dndz[n_excluded_tomo_bins:],
    #         weights=list(tomo_weights.values())[n_excluded_tomo_bins:],
    #         axis=0,
    #     ),
    # )
    #
    # mask = gold_WL["TOMO_BIN"] > 0
    #
    # if n_excluded_tomo_bins:
    #     for exclude_tomo_bin in range(1, n_excluded_tomo_bins + 1):
    #         mask *= gold_WL["TOMO_BIN"] != exclude_tomo_bin
    #
    # source = Source(
    #     ra=gold_WL[mask]["RAJ2000"],
    #     dec=gold_WL[mask]["DECJ2000"],
    #     e1=gold_WL[mask]["e1"],
    #     e2=gold_WL[mask]["e2"],
    #     w=gold_WL[mask]["weight"],
    #     dndz=dndz,
    #     m=float(m),
    # )
    #
    # del gold_WL
    # gc.collect()
    # endregion

    # region Construct source with KiDS-1000 METACALIB WL catalogue
    m_bias = {
        1: -0.013,
        2: -0.010,
        3: -0.011,
        4: 0.007,
        5: 0.006,
    }

    import pyarrow.feather as feather
    table = feather.read_table(
        os.path.join(_KiDS_data, "KiDS_DR4.1_full_WL_cat.ftr"))
    metacal_WL = table.to_pandas()

    n_excluded_tomo_bins = 1

    tomo_weights = {}
    for tomo in range(1, 6):
        tomo_weights[tomo] = float(
            metacal_WL[metacal_WL["TOMO_BIN"] == tomo]["weight"].sum()
        )

    m = np.average(  # compute average m-bias
        list(m_bias.values())[n_excluded_tomo_bins:],
        weights=list(tomo_weights.values())[n_excluded_tomo_bins:],
    )

    dndz = []
    for tomo in range(1, 6):
        tomo_dndz = np.loadtxt(
            os.path.join(
                "/disks/shear16/myoon/K1000_metacal_CS/CosmoWrapper_Outputs",
                f"K1000_NS_DIRcols_Fid_blindNONE_TOMO{tomo:d}_Nz.asc",
            ),
            skiprows=1,
            unpack=True,
        )
        dndz.append(tomo_dndz[1])

    dndz = np.array(dndz)

    dndz = (
        tomo_dndz[0],
        np.average(
            dndz[n_excluded_tomo_bins:],
            weights=list(tomo_weights.values())[n_excluded_tomo_bins:],
            axis=0,
        ),
    )

    mask = metacal_WL["TOMO_BIN"] > 0

    if n_excluded_tomo_bins:
        for exclude_tomo_bin in range(1, n_excluded_tomo_bins + 1):
            mask *= metacal_WL["TOMO_BIN"] != exclude_tomo_bin

    source = Source(
        ra=metacal_WL[mask]["RAJ2000"],
        dec=metacal_WL[mask]["DECJ2000"],
        e1=metacal_WL[mask]["e1"],
        e2=metacal_WL[mask]["e2"],
        w=metacal_WL[mask]["weight"],
        dndz=dndz,
        m=float(m),
    )

    del metacal_WL
    gc.collect()
    # endregion

    # region Construct random catalogues
    randoms = []
    for random_path in os.listdir(os.path.join(_KiDS_data, "randoms")):
        path = os.path.join(_KiDS_data, "randoms", random_path)
        random = Random.from_random_catalogue(path)
        randoms.append(random)
    # endregion

    patch_centers_savepath = (
        "/data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL/"
        "patch_centers_npatch{}.dat"
    )
    npatch = 100

    fname = "dsigma_Lensbin{}_Sourcebin2345.csv"

    for i, (lens, min_rp, max_rp, nbins) in enumerate([
        (lens1, 0.025, 15, 12),
        (lens2, 0.040, 20, 14),
    ], start=1):
        dsigma = DSigma(
            lens=lens,
            source=source,
            randoms=randoms,
            n_rp_bins=nbins,
            min_rp=min_rp,
            max_rp=max_rp,
            var_method="jackknife",
            patch_centers=patch_centers_savepath.format(npatch),
            npatch=npatch,
        )
        dsigma.save(fname.format(i), save_dir=save_dir)
        del dsigma
        gc.collect()