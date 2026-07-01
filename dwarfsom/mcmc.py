import numpy as np
from emcee import EnsembleSampler
from dynesty import NestedSampler, DynamicNestedSampler
from typing import Callable
from multiprocessing import Pool

from dwarfsom.utils import _tqdm_style

class MCMC:
    def __init__(
            self,
            model: Callable,
            data_vector: np.ndarray,
            covariance_matrix: np.ndarray,
            param_priors: dict[str, list[str | tuple[float, float | np.ndarray]]] = None,
    ):
        self.sampler = None
        self.model = model
        self.dv = data_vector
        self.ndim = len(param_priors)

        try:
            np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite.")

        self.cov = covariance_matrix
        self.inv_cov = np.linalg.inv(covariance_matrix)
        self.param_priors = param_priors
        self.param_ranges = [
            prior_range for prior_type, prior_range in param_priors.values()
        ]
        self.n_dim = len(param_priors.keys()) # number of free parameters being sampled


    def _log_prior(self, params):
        """Log prior of the parameters.

        Notes:
            Support only flat priors, ["flat", (low, high)].
        """
        log_prior = 0.

        if self._outside_param_ranges(params):
            return -np.inf

        return log_prior

    def Gaussian_log_likelihood(self, params):
        if self._outside_param_ranges(params):
            return -np.inf
        log_prob = -0.5 * (self.chi2(params)) + self._log_prior(params)
        return log_prob

    def chi2(self, params):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        pred = self.model(params).squeeze()
        chi2 = (self.dv - pred).T @ self.inv_cov @ (self.dv - pred)
        return chi2

    def prior_transform(self, u):
        """Prior transform function for nested sampler."""
        x = np.array(u)
        for i, param_range in enumerate(self.param_ranges):
            low, high = param_range
            x[i] = low + (high - low) * x[i]

        return x


    def get_random_walk(self, n_walkers: int):
        """Generate a random walk for the emcee MCMC sampler."""
        p0 = np.zeros((n_walkers, self.n_dim))

        for i, param in enumerate(self.param_priors.keys()):
            prior_type = self.param_priors[param][0]

            if prior_type == "flat":
                low, high = self.param_priors[param][1]
                p0[:, i] = np.random.uniform(
                    low=low, high=high, size=n_walkers)

            if prior_type == "gaussian":
                mean, cov = self.param_priors[param][1]
                if isinstance(cov, float):
                    p0[:, i] = np.random.normal(
                        loc=mean, scale=cov, size=n_walkers)

        return p0


    def register_sampler(self, sampler):
        self.sampler = sampler

    def get_chain(self, flat=True):
        """Get the chain of samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")

        if isinstance(self.sampler, EnsembleSampler):
            return self.sampler.get_chain(flat=flat)

        return self.sampler.results.samples

    def save_chain(self, flat=True, fname="mcmc_chain.npy"):
        """Save the chain of samples from the MCMC sampler."""
        if isinstance(self.sampler, EnsembleSampler):
            np.save(fname, self.get_chain(flat=flat))

        np.save(fname, self.get_chain())

    def save_log_prob(self, flat=True, fname="mcmc_log_prob.npy"):
        """Save the log probabilities of the samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")

        if isinstance(self.sampler, EnsembleSampler):
            np.save(fname, self.sampler.get_log_prob(flat=flat))

        np.save(fname, self.sampler.results.logl)

    def _outside_param_ranges(self, params):
        """Check if any parameter is outside its allowed range for emcee sampler"""
        for p, (low, high) in zip(params, self.param_ranges):
            if p < low or p > high:
                return True

        return False



def run_emcee(
        mcmc: MCMC,
        n_walkers: int=100,
        n_burn_in_steps: int=100,
        n_steps: int=500,
        processes: int=4,
        progress: bool=True,
        progress_kwargs=_tqdm_style,
):
    with Pool(processes=processes) as pool:
        sampler = EnsembleSampler(
            nwalkers=n_walkers,
            ndim=mcmc.ndim,
            log_prob_fn=mcmc.Gaussian_log_likelihood,
            pool=pool,
        )
        state = sampler.run_mcmc(
            initial_state=mcmc.get_random_walk(n_walkers=n_walkers),
            nsteps=n_burn_in_steps,
            progress=progress,
            progress_kwargs=progress_kwargs,
        )
        sampler.reset()
        sampler.run_mcmc(
            initial_state=state,
            nsteps=n_steps,
            progress=progress,
            progress_kwargs=progress_kwargs,
        )
    return sampler


def run_nested(
        mcmc: MCMC,
        processes: int=4,
        dynamic: bool=False,
        nested_kwargs: dict=None,
):
    """

    Args:
        mcmc:
        processes:
        dynamic:
        nested_kwargs: Keyword argumnts for NestedSampler.run_nested or
            DynamicNestedSampler.run_nested function.
    """
    from dynesty.pool import Pool

    if nested_kwargs is None:
        nested_kwargs = {}

    with Pool(
        processes,
        mcmc.Gaussian_log_likelihood,
        mcmc.prior_transform,
    ) as pool:
        # Restore sampler
        if "resume" in nested_kwargs.keys():
            if nested_kwargs["resume"]:
                if "checkpoint_file" not in nested_kwargs.keys():
                    raise ValueError(
                        "checkpoint_file must be specified when resume=True for nested sampling."
                    )

                if dynamic:
                    sampler = DynamicNestedSampler.restore(
                        nested_kwargs["checkpoint_file"], pool=pool)
                else: # use static nested sampler
                    sampler = NestedSampler.restore(
                        nested_kwargs["checkpoint_file"], pool=pool)
        # Create new sampler
        else:
            if dynamic:
                sampler = DynamicNestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    mcmc.ndim,
                    pool=pool,
                )
            else: # use static nested sampler
                sampler = NestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    mcmc.ndim,
                    pool=pool,
                )

        sampler.run_nested(**nested_kwargs)

    return sampler


def compute_quantiles(
        samples: np.ndarray,
        quantiles: tuple[float]=(0.16, 0.5, 0.84),
) -> list[float]:
    """Compute sample quantiles.

    Args:
        samples: Samples from the posterior distribution. Can be a 1D array of
            samples for a single parameter, or a 2D array of shape
            (n_samples, n_params).
        quantiles:
    """
    if np.ndim(samples) == 1:
        samples = samples[:, np.newaxis] # convert to shape (n_samples, 1)
    quantiles = np.asarray(quantiles)

    qvalues = []

    if not all((0. < quantile < 1.) for quantile in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    for param_samples in samples.T:
        qvalues_i = np.percentile(param_samples, list(100 * quantiles))
        qvalues.append(qvalues_i.tolist())

    if len(qvalues) == 1:
        return qvalues[0]
    else:
        return qvalues


def estimate_MAP(
        samples: np.ndarray,
        method: str="kde",
        log_prob: np.ndarray | None=None,
        kde_kwargs: dict | None=None,
) -> list[float]:
    """Estimate the maximum a posteriori (MAP) estimate from the samples according
    to the log probabilities.
    """
    n_samples, n_params = samples.shape
    match method:
        case "map_sample":
            if log_prob is None:
                raise ValueError("For sample_map method, log_prob must be provided.")
            assert len(samples) == len(log_prob)
            return samples[np.argmax(log_prob)].tolist()

        case "knn":
            from sklearn.neighbors import NearestNeighbors
            k = max(10, int(np.sqrt(n_samples)))
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(samples)
            dist, _ = nbrs.kneighbors(samples)
            dist_safe = dist[:, 1:] + 1.e-10
            dens = np.mean(1. / dist_safe, axis=1)
            return samples[np.argmax(dens)].tolist()

        case "meanshift":
            raise NotImplementedError

        case "kde":
            from seaborn._statistics import KDE
            MAPs = []
            for param_samples in samples.T:
                kde = KDE(**kde_kwargs)
                dens, support = kde(param_samples)
                MAPs.append(float(support[np.argmax(dens)]))
            return MAPs

        case _:
            raise ValueError(f"Unknown method for MAP estimation.")