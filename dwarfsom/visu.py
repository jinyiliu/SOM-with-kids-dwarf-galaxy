import warnings

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmplstyle
from cmplstyle import *
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D

from dwarfsom.mcmc import compute_quantiles, estimate_MAP

cmplstyle.use_builtin_mplstyle()

warnings.filterwarnings(
    action="ignore",
    message="The following kwargs were not used by contour",
    category=UserWarning,
)

CONFIDENCE_LEVELS_2D = (0.393, 0.864)
DEFAULT_FILL_ALPHA = 0.7


def _format_ticks(ticks: tuple[float, ...]) -> list[str]:
    """Format tick values with enough decimals to reflect their spacing."""
    vals = np.sort(np.unique(np.asarray(ticks, dtype=float)))
    if vals.size <= 1:
        decimals = 1
    else:
        step = np.min(np.diff(vals))
        if step <= 0:
            decimals = 1
        else:
            decimals = max(0, int(np.ceil(-np.log10(step))))
    return [f"${v:.{decimals}f}$" for v in ticks]

def corner(
        samples: np.ndarray | list[np.ndarray],
        color: str | list[str]="蔚蓝",
        sample_labels: list[str] | None=None,
        same_contour_color: bool=False,
        fill: bool | list[bool]=False,
        contour_kwargs: dict | list[dict] | None=None,
        fill_kwargs: dict | list[dict] | None=None,
        levels: tuple[float]=CONFIDENCE_LEVELS_2D,
        quantiles: tuple[float]=(0.16, 0.5, 0.84),
        kde_bw_adjust: float=1.0,
        kde_gridsize: int=100,
        param_ranges: list[tuple[float, float]] | None=None,
        param_labels: list[str] | None=None,
        param_ticks: list[tuple[float, ...]] | None=None,
        figsize: tuple[float] | float=median_wth,
        truths: list[float] | None=None,
        truths_kwargs: dict | None=None,
        MAPs: list[float] | None=None,
        MAPs_kwargs: dict | None=None,
        plot_samples: bool=False,
        plot_samples_kwargs: dict | list[dict] | None=None,
        marginal_titles: list[str] | bool=False,
        fill_quantile_band: bool=False,
        verbose: bool=False,
        savedir: str | None=None,
        fname: str="posterior_corner.pdf",
):
    """Plot corner plot of posterior distribution.

    Notes:
        This function uses seaborn.kdeplot to plot the contour of the posterior
        distribution. The confidence levels for the contour are defined in
        CONFIDENCE_LEVELS_2D.

    Args:
        samples: Samples from the posterior distribution. Either a single 2D
            array of shape (n_samples, n_params), or a list of such arrays to
            overlay multiple sample sets in the same panels.
        color: Base color of the contour lines and fill. Either a single color
            (broadcast to all sample sets) or a list of colors, one per set.
        sample_labels: Optional labels for each sample set (for the legend).
            Length must match the number of sample sets.
        same_contour_color: Whether to use the same color for all confidence levels.
        fill: Whether to fill the contour. Either a single bool (broadcast to
            all sample sets) or a list of bools, one per sample set.
        contour_kwargs: Additional keyword arguments for the sns.kdeplot for the
            contour. Either a single dict (broadcast) or a list of dicts, one
            per sample set. May override the defaults for 'levels', 'color',
            'colors', 'bw_adjust', 'gridsize', 'fill', and 'zorder'.
        fill_kwargs: Additional keyword arguments for the sns.kdeplot for the
            filled contour. Either a single dict or a list of dicts, one per
            sample set. May override the defaults for 'levels', 'color',
            'colors', 'bw_adjust', 'gridsize', 'fill', 'alpha', 'extend', and
            'zorder'.
        levels: Confidence levels for the contour.
        quantiles: Quantiles to display on the diagonal plots.
        kde_bw_adjust: Bandwidth adjustment for the kernel density estimation.
            Higher values lead to smoother contours.
        kde_gridsize: Gridsize for the kernel density estimation. Higher values
            lead to smoother contours but longer computation time.
        param_ranges: Parameter ranges for the posterior distribution.
        param_labels: Labels for the parameters. If None, will use the keys
            of param_ranges.
        param_ticks: Ticks for the parameters.
        figsize: Tuple of figure size in cm. If a single float is supplied, it
            will be used for both width and height.
        truths: Optional truth values for the parameters.
        truths_kwargs: Additional keyword arguments for plotting the truth values.
        MAPs: Optional maximum a posteriori (MAP) estimates for the parameters.
        MAPs_kwargs: Additional keyword arguments for plotting the MAP estimates.
        plot_samples: Whether to plot samples as hexbin in the lower triangle.
        plot_samples_kwargs: Additional keyword arguments for the hexbin plot
            of the samples.
        marginal_titles: Whether to add a title to the marginal distribution.
        fill_quantile_band: If True, fill the area between the marginal KDE
            curve and the x-axis within the quantile range, using the same
            color and alpha as the 2sigma fill.
        verbose:
        savedir:
        fname:
    """
    if not all(0. < level < 1. for level in levels):
        raise ValueError("Confidence levels must be between 0 and 1")

    if not all(0. < quantile < 1. for quantile in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    default_truths_kwargs = {
        "color": "丹枫",
        "marker": "x",
        "zorder": 10,
        "label": "Truth",
    }

    if truths:
        if truths_kwargs is None:
            truths_kwargs = default_truths_kwargs
        else:
            truths_kwargs = default_truths_kwargs.update(truths_kwargs)

    default_MAPs_kwargs = {
        "color": "明黄",
        "marker": "x",
        "zorder": 9,
        "label": "MAP",
    }

    if MAPs:
        if MAPs_kwargs is None:
            MAPs_kwargs = default_MAPs_kwargs
        else:
            MAPs_kwargs = default_MAPs_kwargs.update(MAPs_kwargs)

    # Normalize samples into a list of 2D arrays (one per overlaid set).
    if isinstance(samples, (list, tuple)):
        samples_list = [np.asarray(s) for s in samples]
        if truths is not None or MAPs is not None:
            raise ValueError(
                "truths/MAPs are not supported when multiple sample sets "
                "are provided."
            )
        if plot_samples:
            raise ValueError(
                "plot_samples (hexbins) is not supported when multiple "
                "sample sets are provided."
            )
    else:
        samples_list = [np.asarray(samples)]

    n_sets = len(samples_list)
    for s in samples_list:
        if s.ndim != 2:
            raise NotImplementedError(
                "1D samples are not supported."
                "Please provide 2D samples with shape (n_samples, n_params)."
            )
    n_params = samples_list[0].shape[1]
    for s in samples_list:
        if s.shape[1] != n_params:
            raise ValueError(
                "All sample sets must have the same number of parameters."
            )

    # Broadcast per-set scalar/list parameters.
    def _broadcast(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != n_sets:
                raise ValueError(f"{name} must have length {n_sets}.")
            return list(value)
        return [value] * n_sets

    colors = _broadcast(color, "color")
    fills = _broadcast(fill, "fill")


    def _broadcast_kwargs(kwargs, name):
        if kwargs is None:
            return [{} for _ in range(n_sets)]
        if isinstance(kwargs, list):
            if len(kwargs) != n_sets:
                raise ValueError(f"{name} must have length {n_sets}.")
            return [dict(k or {}) for k in kwargs]
        return [dict(kwargs or {}) for _ in range(n_sets)]

    contour_kwargs_per = _broadcast_kwargs(contour_kwargs, "contour_kwargs")
    fill_kwargs_per = _broadcast_kwargs(fill_kwargs, "fill_kwargs")
    plot_samples_kwargs_per = _broadcast_kwargs(
        plot_samples_kwargs, "plot_samples_kwargs")

    if sample_labels is not None and len(sample_labels) != n_sets:
        raise ValueError(f"sample_labels must have length {n_sets}.")

    n_colors = len(CONFIDENCE_LEVELS_2D)

    if param_ranges is None:
        all_samples = np.concatenate(samples_list, axis=0)
        param_ranges = np.vstack(
            [all_samples.min(axis=0), all_samples.max(axis=0)]).T

    if param_ticks is None:
        param_ticks = [
            np.linspace(
                start=minv - (maxv - minv) / 8,
                stop=maxv + (maxv - minv) / 8,
                num=5,
            )[1:-1] for minv, maxv in param_ranges
        ]

    mpl.rcParams["figure.constrained_layout.use"] = False
    mpl.rcParams["xtick.top"] = False
    mpl.rcParams["ytick.right"] = False

    if isinstance(figsize, float):
        figsize = (figsize,) * 2

    fig, axes = plt.subplots(
        figsize=cm2inch(*figsize),
        ncols=n_params,
        nrows=n_params,
    )
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    axes_diag = np.diag(axes)
    axes_lower = np.tril(axes, k=-1)
    axes_upper = np.triu(axes, k=1)

    for ax in axes_upper.flatten():
        if isinstance(ax, plt.Axes):
            ax.axis("off")  # clear upper triangle axes

    for i in range(n_params):  # lower triangle
        for j in range(n_params):
            ax = axes_lower[i, j]
            if isinstance(ax, plt.Axes):
                if plot_samples and n_sets == 1:
                    if param_ranges is not None:
                        extent = (*param_ranges[j], *param_ranges[i])
                    else:
                        extent = None
                    ax.hexbin(
                        x=samples_list[0][:, j],
                        y=samples_list[0][:, i],
                        extent=extent,
                        zorder=0,
                        linewidths=0.05,
                        **plot_samples_kwargs_per[0],
                    )
                levels = [1 - cfl for cfl in CONFIDENCE_LEVELS_2D[::-1]]
                if verbose:
                    print(
                        "KDE plotting for parameters",
                        param_labels[j] if param_labels else f"param_{j}",
                        "and",
                        param_labels[i] if param_labels else f"param_{i}",
                    )
                for s_idx, samp in enumerate(samples_list):
                    color_i = colors[s_idx]
                    palette = sns.light_palette(
                        color=color_i, n_colors=n_colors + 2)[-n_colors:]
                    ck = dict(contour_kwargs_per[s_idx])
                    sns.kdeplot( # posterior contour plot
                        x=samp[:, j],
                        y=samp[:, i],
                        ax=ax,
                        levels=ck.pop("levels", levels),
                        color=ck.pop("color", color_i),
                        bw_adjust=ck.pop("bw_adjust", kde_bw_adjust),
                        gridsize=ck.pop("gridsize", kde_gridsize),
                        fill=ck.pop("fill", False),
                        colors=ck.pop(
                            "colors",
                            [color_i] * len(levels) if same_contour_color
                            else palette,
                        ),
                        zorder=ck.pop("zorder", 2 + s_idx),
                        **ck,
                    )
                    if fills[s_idx]:
                        fk = dict(fill_kwargs_per[s_idx])
                        alpha = fk.pop("alpha", DEFAULT_FILL_ALPHA)
                        n_levels = len(levels)
                        # 2sigma (outer) fill -> alpha; 1sigma (inner) -> alpha + 0.2
                        alphas = [alpha] * n_levels
                        alphas[-1] = min(alpha + 0.2, 1.0)
                        sns.kdeplot(
                            x=samp[:, j],
                            y=samp[:, i],
                            ax=ax,
                            levels=fk.pop("levels", levels),
                            color=fk.pop("color", color_i),
                            bw_adjust=fk.pop("bw_adjust", kde_bw_adjust),
                            gridsize=fk.pop("gridsize", kde_gridsize),
                            fill=fk.pop("fill", True), # use matplotlib.axes.Axes.contourf
                            colors=fk.pop("colors", palette),
                            alpha=alphas,
                            extend=fk.pop("extend", "max"),
                            zorder=fk.pop("zorder", 1 + s_idx),
                            **fk,
                        )
                ax.set_xlim(param_ranges[j])
                ax.set_ylim(param_ranges[i])
                ax.set_xticks(param_ticks[j])
                ax.set_xticklabels([])
                ax.set_yticks(param_ticks[i])
                ax.set_yticklabels([])

                if n_sets == 1:
                    if truths is not None:
                        ax.scatter(
                            truths[j], truths[i],
                            **(
                                truths_kwargs if (i, j) == (n_params - 1, 0) else
                                {k: v for k, v in truths_kwargs.items() if (k != "label")}
                            ),
                        )

                    if MAPs is not None:
                        ax.scatter(
                            MAPs[j], MAPs[i],
                            **(
                                MAPs_kwargs if (i, j) == (n_params - 1, 0) else
                                {k: v for k, v in MAPs_kwargs.items() if (k != "label")}
                            ),
                        )

    if n_sets > 1:
        handles = [
            Line2D(
                xdata=[0], ydata=[0],
                color=colors[s_idx],
                label=(
                    sample_labels[s_idx] if sample_labels is not None
                    else f"set {s_idx}"
                ),
            )
            for s_idx in range(n_sets)
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            bbox_transform=axes[0, -1].transAxes,
            prop={"family": "sans serif"},
            frameon=False,
        )
    else:
        handles = [
            h for ax in axes.flat
            for h in ax.get_legend_handles_labels()[0]
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=axes[0, -1].transAxes,
                prop={"family": "sans serif"},
                frameon=False,
            )

    qvalues_list = [
        compute_quantiles(samp, quantiles=quantiles) for samp in samples_list
    ]

    for i, ax in enumerate(axes_diag):  # diagonal
        if isinstance(ax, plt.Axes):
            for s_idx, samp in enumerate(samples_list):
                sns.kdeplot(
                    samp[:, i],
                    ax=ax,
                    bw_adjust=kde_bw_adjust,
                    gridsize=kde_gridsize,
                    color=colors[s_idx],
                    linewidth=1.6,
                    zorder=0,
                )
                if fill_quantile_band:
                    qlow, _, qhigh = qvalues_list[s_idx][i]
                    line = ax.get_lines()[-1]
                    xs = line.get_xdata()
                    ys = line.get_ydata()
                    palette = sns.light_palette(
                        color=colors[s_idx], n_colors=n_colors + 2)[-n_colors:]
                    band_alpha = fill_kwargs_per[s_idx].get(
                        "alpha", DEFAULT_FILL_ALPHA)
                    mask = (xs >= qlow) & (xs <= qhigh)
                    xs_m = np.where(mask, xs, np.nan)
                    ys_m = np.where(mask, ys, np.nan)
                    ax.fill_between(
                        xs_m, ys_m, 0,
                        color=palette[0],
                        alpha=band_alpha,
                        linewidth=0,
                        edgecolor="none",
                        zorder=-1,
                    )
            ax.margins(y=0.1)
            ax.set_ylim(bottom=0.0)

            if n_sets == 1:
                qlow, qmid, qhigh = qvalues_list[0][i]
                if verbose:
                    print(
                        param_labels[i] if param_labels else f"param_{i}",
                        f"= {qmid:.3f} {{+{qmid - qlow:.3f}}} {{-{qhigh - qmid:.3f}}}",
                    )
                if marginal_titles:
                    if isinstance(marginal_titles, list):
                        ax.set_title(
                            label=marginal_titles[i],
                            fontsize=7,
                        )
                    else:
                        ax.set_title(
                            label=param_labels[i] + f" $={qmid:.2f}^{{+{qmid - qlow:.2f}}}_{{-{qhigh - qmid:.2f}}}$",
                            fontsize=7,
                        )

            ax.set_xlim(param_ranges[i])
            ax.set_xticks(param_ticks[i])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if n_sets == 1 and truths is not None:
                ax.axvline(
                    truths[i],
                    color=truths_kwargs["color"],
                    linewidth=0.8,
                    zorder=truths_kwargs["zorder"],
                )

            if n_sets == 1 and MAPs is not None:
                ax.axvline(
                    MAPs[i],
                    color=MAPs_kwargs["color"],
                    linewidth=0.8,
                    zorder=MAPs_kwargs["zorder"],
                )

    for ax in axes.flatten():
        ax.set_xlabel("")  # clear x-axis labels
        ax.set_ylabel("")

    # Add x-tick labels and y-axis labels for the bottom row
    for i, ax in enumerate(axes[-1, :]):
        if param_labels is not None:
            ax.set_xlabel(param_labels[i])
        if param_ticks is not None:
            ax.set_xticklabels(
                _format_ticks(param_ticks[i]),
                fontsize=8,
            )

    # Add y-tick labels and y-axis labels for the leftmost column
    for i, ax in enumerate(axes[:, 0]):
        if i != 0:
            if param_labels is not None:
                ax.set_ylabel(param_labels[i])
            if param_ticks is not None:
                ax.set_yticklabels(
                    _format_ticks(param_ticks[i]),
                    fontsize=8,
                )

    if savedir:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )

    return fig, axes


def overlay_line(
        x, y,
        x0=None,
        bg_color="white",
        n_colors=256,
        linewidth=2.,
):
    x = np.asarray(x)
    y = np.asarray(y)

    if x0 is None:
        x0 = np.median(x)

    width = 0.05
    def smooth_step(x_val):
        return 1 / (1 + np.exp(-(x_val - x0) / width))

    # Resample onto a dense uniform x-grid so the curve is a smooth stroke
    # rather than a visibly beaded chain of segments.
    x_dense = np.linspace(x.min(), x.max(), n_colors * 4)
    y_dense = np.interp(x_dense, x, y)
    x, y = x_dense, y_dense
    transition_values = smooth_step(x)
    bg_rgb = to_rgba(bg_color)

    # Colormap from fully opaque bg_color to fully transparent bg_color.
    colors_list = [bg_rgb, (*bg_rgb[:3], 0.0)]
    cmap = LinearSegmentedColormap.from_list(
        name="fade", colors=colors_list, N=n_colors)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(0, 1),
        linewidth=linewidth,
    )
    lc.set_array(transition_values[:-1])

    return lc


