import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle as pk
import cmplstyle
from tqdm import tqdm
from cmplstyle.utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cmplstyle.use_builtin_mplstyle()

_alblas = "/net/alblas"
sys.path.append(os.path.join(
    _alblas,
    "data1/jliu/SOM-with-kids-dwarf-galaxy",
))

from dwarfsom.visu import corner
from dwarfsom.model import (
    DSigmaModel,
    DSigma_1h_sm_host,
    precompute_host_dsigma_terms,
)
from dwarfsom.dsigma import DSigmaData
from dwarfsom.mcmc import (
    MCMC, run_emcee, run_nested, estimate_MAP, compute_quantiles, _tqdm_style
)


# region Ignore warnings
warnings.filterwarnings(
    action="ignore", message="There are no gridspecs with layoutgrids.*",
    category=UserWarning,
)
warnings.filterwarnings(
    action="ignore", message="The figure layout has changed to tight*",
    category=UserWarning,
)
# endregion


_SOM_data = "data1/jliu/SOM-with-kids-dwarf-galaxy/data/SOM"
_GGL_data = "data1/jliu/SOM-with-kids-dwarf-galaxy/data/GGL"
fname = "dsigma_Lensbin{}_Sourcebin2345.csv"

dndlogmstar = pd.read_csv(
    os.path.join(_alblas, _SOM_data, "dndlogmstar_by_mass_bins.csv"))


# region Load data: cov1, cov2, dsigma1, dsigma2
cov1 = np.load(
    os.path.join(
        _alblas,
        _GGL_data,
        fname.format(1).replace(".csv", "_cov.npy")
    )
)
cov2 = np.load(
    os.path.join(
        _alblas,
        _GGL_data,
        fname.format(2).replace(".csv", "_cov.npy")
    )
)

dsigma1 = DSigmaData.load_csv(
    fname=fname.format(1),
    save_dir=os.path.join(
        _alblas,
        _GGL_data,
    )
)
dsigma2 = DSigmaData.load_csv(
    fname=fname.format(2),
    save_dir=os.path.join(
        _alblas,
        _GGL_data,
    ),
)

# endregion

z_lens1 = 0.07275
z_lens2 = 0.11775

M_star1 = 9.153
M_star2 = 9.596


# region Recache host terms in the model if needed
# _cache_host_terms = {}
# for key in [
#     (0.1, None, 1., 9.5),  # for testing
#     (z_lens1, None, 1., M_star1),
#     (z_lens2, None, 1., M_star2),
# ]:
#     _cache_host_terms[key] = precompute_host_dsigma_terms(*key)
#
# with open(
#     os.path.join(
#         _alblas, _GGL_data, "_cache_host_terms.pk"
#     ), "wb"
# ) as f:
#     pk.dump(_cache_host_terms, f)
# endregion

param_priors = {
    "log10_M":      ["flat",    (10.0, 12.0)],
    "R":            ["flat",    (0.0, 2.0)],
    "frac_sat":     ["flat",    (0.0, 1.0)],
}

param_labels = [
    r"$\log_{10}(M_\mathrm{h}/M_\odot)$",
    r"$\mathcal{R}$",
    r"$f_\mathrm{sat}$",
]

n_outliers = 2  # remove the last two data points


# region Model specification
def model1(params: np.ndarray[float]):
    return DSigmaModel(
        rp=dsigma1.mean_rp[:-n_outliers],
        z_lens=z_lens1,
        log10_M=params[0],
        log10_M_star=M_star1,
        log10_M_host=None,
        f_c=1.,
        R=params[1],
        frac_sat=params[2],
    )

def model2(params: np.ndarray[float]):
    return DSigmaModel(
        rp=dsigma2.mean_rp[:-n_outliers],
        z_lens=z_lens2,
        log10_M=params[0],
        log10_M_star=M_star2,
        log10_M_host=None,
        f_c=1.,
        R=params[1],
        frac_sat=params[2],
    )

# endregion


# region MCMC instances
mcmc1 = MCMC(
    model=model1,
    data_vector=dsigma1.dsigma_tangential[:-n_outliers],
    covariance_matrix=cov1[:-n_outliers, :-n_outliers],
    param_priors=param_priors,
)

mcmc2 = MCMC(
    model=model2,
    data_vector=dsigma2.dsigma_tangential[:-n_outliers],
    covariance_matrix=cov2[:-n_outliers, :-n_outliers],
    param_priors=param_priors,
)
# endregion


if __name__ == "__main__":
    # region Run MCMC
    n_walkers = 32
    n_burn_in_steps = 3000
    n_steps = 20000
    processes = 32

    sampler1 = run_emcee(
        mcmc1,
        n_walkers=n_walkers,
        n_burn_in_steps=n_burn_in_steps,
        n_steps=n_steps,
        processes=processes,
    )

    sampler2 = run_emcee(
        mcmc2,
        n_walkers=n_walkers,
        n_burn_in_steps=n_burn_in_steps,
        n_steps=n_steps,
        processes=processes,
    )


    mcmc1.register_sampler(sampler1)
    mcmc2.register_sampler(sampler2)

    mcmc1.save_chain(fname=os.path.join(
        _alblas, _GGL_data, "mcmc_chain_Lensbin1.npy"
    ))
    mcmc2.save_chain(fname=os.path.join(
        _alblas, _GGL_data, "mcmc_chain_Lensbin2.npy"
    ))
    # endregion

    samples1 = np.load(os.path.join(
        _alblas, _GGL_data, "mcmc_chain_Lensbin1.npy"
    ))
    samples2 = np.load(os.path.join(
        _alblas, _GGL_data, "mcmc_chain_Lensbin2.npy"
    ))
    colors = ["石榴裙", "柳绿"]
    labels = [
        r"$\langle \log_{10}(M_\star/M_\odot) \rangle = 8.91$",
        r"$\langle \log_{10}(M_\star/M_\odot) \rangle = 9.50$",
    ]

    # region Plot posterior
    MAPs1 = estimate_MAP(
        samples1,
        method="kde",
        kde_kwargs={
            "bw_adjust": 1.3,
            "gridsize": 70,
        },
    )

    MAPs2 = estimate_MAP(
        samples2,
        method="kde",
        kde_kwargs={
            "bw_adjust": 1.3,
            "gridsize": 70,
        },
    )

    for bin_name, samples, MAPs, mcmc in [
        ("Lens 1", samples1, MAPs1, mcmc1),
        ("Lens 2", samples2, MAPs2, mcmc2),
    ]:
        print(f"\n{bin_name}")
        means = samples.mean(axis=0)
        qvalues = compute_quantiles(samples)
        for param_name, mean, map_val, (qlow, qmid, qhigh) in zip(
            param_priors.keys(), means, MAPs, qvalues,
        ):
            print(
                f"  {param_name}: \n"
                f"      MAP = {map_val:.4f}, \n"
                f"      mean = {mean:.4f}, \n"
                f"      median = {qmid:.4f} "
                f"+{qhigh - qmid:.4f} -{qmid - qlow:.4f}"
            )
        chi2 = mcmc.chi2(MAPs)
        dof = len(mcmc.dv) - len(param_priors)
        chi2_nu = chi2 / dof
        print(
            f"  chi2_nu = {chi2_nu:.3f}  "
            f"(chi2 = {chi2:.2f}, dof = {dof}, evaluated at MAP)"
        )


    plot_samples_kwargs = dict(
        gridsize=30,
        cmap="Greys",
    )
    param_ranges = [
        (10.7, 11.8),
        (0.0, 2.0),
        (0.0, 0.17),
    ]
    param_ticks = [
        (10.9, 11.2, 11.5),
        (0.5, 1.0, 1.5),
        (0.05, 0.10, 0.15),
    ]

    fig, ax = corner(
        samples=[samples1, samples2],
        color=colors,
        sample_labels=labels,
        kde_bw_adjust=2.,
        contour_kwargs={
            "linewidth": 0.1,
        },
        fill=True,
        fill_quantile_band=False,
        fill_kwargs={
            "alpha": 0.7,
        },
        param_labels=param_labels,
        param_ranges=param_ranges,
        param_ticks=param_ticks,
        figsize=onecol_wth,
        MAPs=None,
        savedir="./",
        plot_samples=False,
        plot_samples_kwargs=plot_samples_kwargs,
        marginal_titles=True,
        fname="posterior_corner.pdf",
    )
    # endregion


    # region Plot the GGL measurements together with best-fit model
    sep = np.logspace(-2, 2, 100)
    rng = np.random.default_rng(seed=77)
    n_spaghetti = min(len(samples1), 300)
    n_stats = min(len(samples1), 1000)

    def model1(samples: np.ndarray[float]):
        ret = {
            "total": np.zeros(shape=(len(samples), len(sep))),
            "star": np.zeros(shape=(len(samples), len(sep))),
            "1h_cm": np.zeros(shape=(len(samples), len(sep))),
            "1h_sm_sub": np.zeros(shape=(len(samples), len(sep))),
            "1h_sm_host": np.zeros(shape=(len(samples), len(sep))),
            "2h": np.zeros(shape=(len(samples), len(sep))),
        }
        for i, params in enumerate(samples):
            _, ds = DSigmaModel(
                rp=sep,
                z_lens=z_lens1,
                log10_M=params[0],
                log10_M_star=M_star1,
                R=params[1],
                f_c=1.,
                frac_sat=params[2],
                return_components=True,
            )
            for key in ret.keys():
                ret[key][i] = ds[key]
        return ret

    def model2(samples: np.ndarray[float]):
        ret = {
            "total": np.zeros(shape=(len(samples), len(sep))),
            "star": np.zeros(shape=(len(samples), len(sep))),
            "1h_cm": np.zeros(shape=(len(samples), len(sep))),
            "1h_sm_sub": np.zeros(shape=(len(samples), len(sep))),
            "1h_sm_host": np.zeros(shape=(len(samples), len(sep))),
            "2h": np.zeros(shape=(len(samples), len(sep))),
        }
        for i, params in enumerate(samples):
            _, ds = DSigmaModel(
                rp=sep,
                z_lens=z_lens2,
                log10_M=params[0],
                log10_M_star=M_star2,
                R=params[1],
                f_c=1.,
                frac_sat=params[2],
                return_components=True,
            )
            for key in ret.keys():
                ret[key][i] = ds[key]

        return ret


    idx_spaghetti = rng.choice(len(samples1), n_spaghetti, replace=False)
    components_spag1 = model1(samples1[idx_spaghetti])
    components_spag2 = model2(samples2[idx_spaghetti])
    idx_stats = rng.choice(len(samples1), n_stats, replace=False)
    components_stat1 = model1(samples1[idx_stats])
    components_stat2 = model2(samples2[idx_stats])

    fig = plt.figure(
        figsize=cm2inch(fullpg_wth, fullpg_wth * 0.3),
    )
    fig.tight_layout(pad=0.5, h_pad=0., w_pad=0.)

    outer = GridSpec(
        1, 2,
        wspace=0.23,
        width_ratios=[0.9, 2],
        figure=fig,
    )
    ax0 = fig.add_subplot(outer[0])

    inner = outer[1].subgridspec(1, 2, wspace=0., hspace=0.)
    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1], sharey=ax1)

    for bin_, (
            ax, dsigma, components_spag, components_stat, color, label
    ) in enumerate(zip(
            [ax1, ax2],
            [dsigma1, dsigma2],
            [components_spag1, components_spag2],
            [components_stat1, components_stat2],
            colors,
            labels,
    ), start=1):
        ax0.plot(
            dndlogmstar["BIN_CENTRE"],
            dndlogmstar[f"MASS_BIN_{bin_}"],
            label=label,
            color=color,
            linewidth=2.2,
            zorder=bin_,
        )
        ax0.fill_between(
            dndlogmstar["BIN_CENTRE"],
            dndlogmstar[f"MASS_BIN_{bin_}"], -1,
            color=color,
            alpha=0.3,
            zorder=bin_ + 0.5,
        )

        for line in components_spag["total"]:
            ax.plot(
                sep, line,
                color="牛绒",
                linewidth=0.5,
                linestyle="-",
                zorder=1,
                alpha=0.4,
            )
        median = np.median(components_stat["total"], axis=0)
        low68 = np.percentile(components_stat["total"], q=16, axis=0)
        high68 = np.percentile(components_stat["total"], q=84, axis=0)
        ax.plot(
            sep, median,
            color="元青",
            linewidth=1.3,
            linestyle="-",
            zorder=1.5,
        )
        ax.plot(
            sep, low68,
            color="元青",
            linewidth=0.6,
            linestyle=(5, (10, 3)),
            zorder=1.5,
        )
        ax.plot(
            sep, high68,
            color="元青",
            linewidth=0.6,
            linestyle=(5, (10, 3)),
            zorder=1.5,
        )

        ax.errorbar(
            x=dsigma.mean_rp[:-n_outliers],
            y=dsigma.dsigma_tangential[:-n_outliers],
            yerr=dsigma.dsigma_stderr[:-n_outliers],
            color=color,
            marker=".",
            markersize=6,
            markeredgecolor="青骊",
            markeredgewidth=0.7,
            linestyle="none",
            label=label,
            elinewidth=1.5,
            capsize=2.5,
        )
        ax.errorbar(
            x=dsigma.mean_rp[-n_outliers:],
            y=dsigma.dsigma_tangential[-n_outliers:],
            yerr=dsigma.dsigma_stderr[-n_outliers:],
            markerfacecolor="white",
            ecolor=color,
            marker=".",
            markersize=6,
            markeredgecolor="青骊",
            markeredgewidth=0.7,
            linestyle="none",
            elinewidth=1.5,
            capsize=2.5,
        )
        ax.loglog()
        ax.set_xlim(left=0.02, right=25)
        ax.set_ylim(bottom=0.05, top=35)
        ax.set_xticks(ticks=[0.1, 1, 10], labels=["0.1", "1", "10"])
        ax.set_xlabel(r"$r_\mathrm{p}$ [$h^{-1} \mathrm{Mpc}$]")

    ax0.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax0.set_ylabel(r"PDF")
    ax0.set_xlim(left=6.5, right=10.7)
    ax0.set_ylim(bottom=0., top=2.0)
    ax0.legend(frameon=False)

    ax1.set_yticks(
        ticks=[0.1, 1, 10],
        labels=["0.1", "1", "10"],
    )
    ax2.tick_params(labelleft=False)
    ax1.set_ylabel(
        r"$\Delta\Sigma$ [$h$ $M_\odot\ \mathrm{pc^{-2}}$]"
    )

    fig.savefig(
        "dsigma_lens_source_bins.pdf",
        bbox_inches="tight",
        dpi=800,
    )

    # endregion

