import numpy as np
from emcee import EnsembleSampler
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
            processes: int=4,
    ):
        self.sampler = None
        self.model = model
        self.dv = data_vector
        self.processes = processes

        try:
            np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite.")

        self.cov = covariance_matrix
        self.inv_cov = np.linalg.inv(covariance_matrix)
        self.param_priors = param_priors
        self.n_dim = len(param_priors.keys()) # number of free parameters being sampled


    def _log_prior(self, params):
        """Log prior of the parameters.

        Notes:
            Support flat priors, ["flat", (low, high)], and Gausssian priors,
            ["gaussian", (mean, std)].
        """
        log_prior = 0.

        for param, value in zip(self.param_priors.keys(), params):
            prior_type = self.param_priors[param][0]

            if prior_type == "flat":
                low, high = self.param_priors[param][1]
                if value < low or value > high:
                    return -np.inf

            if prior_type == "gaussian":
                mean, cov = self.param_priors[param][1]
                if isinstance(cov, float):
                    log_prior += -0.5 * ((value - mean) / cov) ** 2
                else:
                    pass

        return log_prior

    def _Gaussian_log_likelihood(self, params):
        return -0.5 * (self.chi2(params)) + self._log_prior(params)

    def chi2(self, params):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        pred = self.model(params).squeeze()
        chi2 = (self.dv - pred).T @ self.inv_cov @ (self.dv - pred)
        return chi2

    def _get_random_walk(self, n_walkers: int):
        """Generate a random walk for the MCMC."""
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

    def run_mcmc(
            self,
            n_walkers: int=100,
            n_steps: int=500,
            n_burn_in_steps: int=100,
            progress: bool=True,
            progress_kwargs=_tqdm_style,
            **kwargs,
    ):
        """Run the MCMC sampler.

        Args:
            n_walkers: Number of walkers in the MCMC sampler.
            n_steps: Number of steps in the MCMC sampler.
            n_burn_in_steps: Number of burn-in steps to discard.
            progress: Show a progress bar using tqdm.
            progress_kwargs: Keyword arguments passed to tqdm.
            kwargs: Additional keyword arguments for the
                emcee.EnsembleSampler.sample method.
        """
        p0 = self._get_random_walk(n_walkers=n_walkers)

        with Pool(processes=self.processes) as pool:
            sampler = EnsembleSampler(
                n_walkers, self.n_dim, self._Gaussian_log_likelihood, pool=pool,
            )
            state = sampler.run_mcmc(
                initial_state=p0,
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
                **kwargs
            )

        self.sampler = sampler

    def get_chain(self, flat=True):
        """Get the chain of samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")

        return self.sampler.get_chain(flat=flat)

    def save_chain(self, flat=True, fname="mcmc_chain.npy"):
        """Save the chain of samples from the MCMC sampler."""
        np.save(fname, self.sampler.get_chain(flat=flat))

    def save_log_prob(self, flat=True, fname="mcmc_log_prob.npy"):
        """Save the log probabilities of the samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")

        np.save(fname, self.sampler.get_log_prob(flat=flat))


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