"""
This script is based on pk_to_real.py in the following repo:
https://github.com/andrejdvornik/onepower_app
"""
import numpy as np
import scipy.interpolate
from ._fftlog import *

# Adapted from CosmoSIS Standard Library module, including the _fftlog.py

TRANSFORM_WP = "wp"
TRANSFORM_DS = "ds"
TRANSFORM_W = "wtheta"
TRANSFORM_XI = "xi"
TRANSFORM_XIP = "xip"
TRANSFORM_XIM = "xim"
TRANSFORM_GAMMA = "gamma"

DEFAULT_N_TRANSFORM = 8192
DEFAULT_K_MIN = 0.0001
DEFAULT_K_MAX = 5.0e6
DEFAULT_RP_MIN = 0.1
DEFAULT_RP_MAX = 1000.0


TRANSFORMS = [
    TRANSFORM_WP,
    TRANSFORM_DS,
    TRANSFORM_W,
    TRANSFORM_XIP,
    TRANSFORM_XIM,
    TRANSFORM_GAMMA,
]

# Bias q and order mu parameters for transform
_TRANSFORM_PARAMETERS = {
    TRANSFORM_WP: (0.0, 0.0),
    TRANSFORM_DS: (0.0, 2.0),
    TRANSFORM_W: (0.0, 0.0),
    TRANSFORM_XIP: (0.0, 0.0),
    TRANSFORM_XIM: (0.0, 4.0),
    TRANSFORM_GAMMA: (0.0, 2.0),
}


class LogInterp:
    """
    This is a helper object that interpolates into f(x) where x>0. If all f>0
    then it interpolates log(f) vs log(x). If they are all f<0 then it
    interpolate log(-f) vs log(x). If f is mixed or has some f=0 then it just
    interpolates f vs log(x).
    """
    def __init__(self, angle, spec, kind):
        if np.all(spec > 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle),
                np.log(spec),
                kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.interp_type = "loglog"
        elif np.all(spec < 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle),
                np.log(-spec),
                kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.interp_type = "minus_loglog"
        else:
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), spec, kind, bounds_error=False, fill_value="extrapolate"
            )
            self.interp_type = "log_ang"

    def __call__(self, angle):
        if self.interp_type == "loglog":
            spec = np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type == "minus_loglog":
            spec = -np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type == "log_ang"
            spec = self.interp_func(np.log(angle))
        return spec


class Transformer:
    """
    Class to build Hankel Transformers that convert from 3D power spectra to
    correlation functions. Several transform types are allowed, depending on
    whether you are using cosmic shear, clustering, or galaxy-galaxy lensing.
    """
    def __init__(
        self, transform_type, n, k_min, k_max, sep_min, sep_max, lower=1.0, upper=-2.0
    ):
        # We use a fixed ell grid in log space and will interpolate/extrapolate our inputs onto this
        # grid. We typically use a maximum ell very much higher than the range we have physical values
        # for.  The exact values there do not matter, but they must be not have a sharp cut-off to avoid
        # oscillations at small angle.
        self.k_min = k_min
        self.k_max = k_max
        k = np.logspace(np.log10(k_min), np.log10(k_max), n)
        self.k = k
        dlogr = np.log(k[1]) - np.log(k[0])

        # _fftlog has several options about how the theta and ell values used are chosen.
        # This option tells it to pick them to minimize ringing.
        kropt = 1

        # The parameters of the Hankel transform depend on the type.
        # They are defined in a dict at the top of the file
        self.q, self.mu = _TRANSFORM_PARAMETERS[transform_type]

        # Prepare the Hankel transform.
        self.kr, self.xsave = fhti(n, self.mu, dlogr, q=self.q, kropt=kropt)

        # We always to the inverse transform, from Fourier->Real.
        self.direction = -1

        # Some more fixed values.
        self.sep_min = sep_min
        self.sep_max = sep_max
        self.lower = lower
        self.upper = upper

        # work out the effective rp values.
        nc = 0.5 * (n + 1)
        log_kmin = np.log(k_min)
        log_kmax = np.log(k_max)
        log_kmid = 0.5 * (log_kmin + log_kmax)
        k_mid = np.exp(log_kmid)
        r_mid = self.kr / k_mid
        x = np.arange(n)

        # And the effective separations of the output
        self.sep = np.exp((x - nc) * dlogr) * r_mid
        # self.rp = np.degrees(self.rp_rad) * 60.0
        self.range = (self.sep > self.sep_min) & (self.sep < self.sep_max)

    def __call__(self, k_in, pk_in, chi_l=None):
        """Convert the input k and P(k) points to the points this transform requires, and then
        transform."""
        # Sample onto self.ell
        pk = self._interpolate_and_extrapolate_pk(k_in, pk_in)

        if self.q == 0:
            xi = (
                fht(self.k * pk, self.xsave, tdir=self.direction)
                / (2 * np.pi)
                / self.sep
            )
        else:
            xi = (
                fhtq(self.k * pk, self.xsave, tdir=self.direction)
                / (2 * np.pi)
                / self.sep
            )

        return self.sep[self.range], xi[self.range]

    def _interpolate_and_extrapolate_pk(self, k, pk):
        """Extrapolate and interpolate the input ell and cl to the default
        points for this transform"""
        k_min = k[0]
        k_max = k[-1]
        interpolator = LogInterp(k, pk, "linear")
        pk_out = interpolator(self.k)
        # bad_low = np.isnan(pk_out) & (self.k < k_min)
        # bad_high = np.isnan(pk_out) & (self.k > k_max)

        # pk_out[bad_low] = pk[0] * (self.k[bad_low] / k_min)**self.lower
        # pk_out[bad_high] = pk[-1] * (self.k[bad_high] / k_max)**self.upper

        return pk_out


class PkTransformer(Transformer):
    def __init__(
        self,
        corr_type,
        model,
        k_min=None,
        k_max=None,
        sep_min_in=None,
        sep_max_in=None,
        n_transform=None,
        z_s=None,
        components=False,
    ):
        self.corr_type = corr_type
        self.model = model

        if k_min is None:
            k_min = DEFAULT_K_MIN
        if k_max is None:
            k_max = DEFAULT_K_MAX
        if sep_min_in is None:
            sep_min_in = DEFAULT_RP_MIN
        if sep_max_in is None:
            sep_max_in = DEFAULT_RP_MAX
        if n_transform is None:
            n_transform = DEFAULT_N_TRANSFORM

        self.k_vec = self.model.k_vec  # h/Mpc
        self.z_l = self.model.z_vec[0]
        self.z_s = z_s
        self.components = components

        # Cosmology
        cosmo = self.model.cosmo_model
        self.h = cosmo.h
        self.H0 = cosmo.H0.value  # km/s/Mpc
        self.Omega_m = cosmo.Om0

        # Comoving distance
        chi_Mpc = cosmo.comoving_distance(self.z_l + 1e-6).to("Mpc").value
        self.chi_l_Mpc = chi_Mpc  # Mpc (for lensing kernel)
        self.chi_l = chi_Mpc * self.h  # Mpc/h (for Hankel transform)

        # Separation limits
        if corr_type in ["ds", "wp"]:
            sep_min = sep_min_in
            sep_max = sep_max_in
        else:
            conv = ( # convert arcmin → Mpc/h
                cosmo.kpc_comoving_per_arcmin(self.z_l + 1e-6).to("Mpc / arcmin").value
            )

            sep_min = sep_min_in * conv * self.h
            sep_max = sep_max_in * conv * self.h

        self.n = n_transform

        super().__init__(corr_type, self.n, k_min, k_max, sep_min, sep_max)


    def __call__(self):
        if self.corr_type == "ds":
            P_list = []
            P_1h_list = []
            P_2h_list = []
            for i in range(self.model.power_spectrum_gm.pk_tot.shape[0]):
                P_list.append(self.model.power_spectrum_gm.pk_tot[i, i, :])
                P_1h_list.append(self.model.power_spectrum_gm.pk_1h[i, i, :])
                P_2h_list.append(self.model.power_spectrum_gm.pk_2h[i, i, :])

        elif self.corr_type == "wp" or self.corr_type == "wtheta":
            P = self.model.power_spectrum_gg.pk_tot[0, 0, :]
            if self.components:
                P_1h = self.model.power_spectrum_gg.pk_1h[0, 0, :]
                P_2h = self.model.power_spectrum_gg.pk_2h[0, 0, :]

        elif self.corr_type in ["gamma", "xip", "xim"]:
            c = 299792.458
            a_l = 1.0 / (1.0 + self.z_l)

            # Kernel W in 1/Mpc
            if self.z_s is None:
                W = (3 * self.H0**2 * self.Omega_m / (2 * c**2)) * (
                    self.chi_l_Mpc / a_l
                )
            else:
                chi_s_Mpc = (
                    self.model.cosmo_model.comoving_distance(self.z_s).to("Mpc").value
                )

                W = (
                    (3 * self.H0**2 * self.Omega_m / (2 * c**2))
                    * (self.chi_l_Mpc / a_l)
                    * ((chi_s_Mpc - self.chi_l_Mpc) / chi_s_Mpc)
                )

            if self.corr_type == "gamma":
                P = W * (1.0 / self.h) * self.model.power_spectrum_gm.pk_tot[0, 0, :]
                if self.components:
                    P_1h = (
                        W * (1.0 / self.h) * self.model.power_spectrum_gm.pk_1h[0, 0, :]
                    )
                    P_2h = (
                        W * (1.0 / self.h) * self.model.power_spectrum_gm.pk_2h[0, 0, :]
                    )

            else:  # xi+ or xi-
                P = W**2 * (1.0 / self.h) * self.model.power_spectrum_mm.pk_tot[0, 0, :]
                if self.components:
                    P_1h = (
                        W**2
                        * (1.0 / self.h)
                        * self.model.power_spectrum_mm.pk_1h[0, 0, :]
                    )
                    P_2h = (
                        W**2
                        * (1.0 / self.h)
                        * self.model.power_spectrum_mm.pk_2h[0, 0, :]
                    )
        else:
            raise ValueError("Unknown transform type")

        if self.corr_type == "ds":
            xi_list = []
            xi_1h_list = []
            xi_2h_list = []
            for P, P_1h, P_2h in zip(P_list, P_1h_list, P_2h_list):
                sep_out, xi = super().__call__(self.k_vec, P)
                xi *= self.model.mean_density0[0]  # M_sun / (Mpc/h)^3
                xi /= 1e12  # M_sun / pc^2
                xi_list.append(xi)
                if self.components:
                    _, xi_1h = super().__call__(self.k_vec, P_1h)
                    _, xi_2h = super().__call__(self.k_vec, P_2h)
                    xi_1h *= self.model.mean_density0[0]  # M_sun / (Mpc/h)^3
                    xi_1h /= 1e12  # M_sun / pc^2
                    xi_2h *= self.model.mean_density0[0]  # M_sun / (Mpc/h)^3
                    xi_2h /= 1e12  # M_sun / pc^2

                    xi_1h_list.append(xi_1h)
                    xi_2h_list.append(xi_2h)


        if self.corr_type in ["wtheta", "gamma", "xip", "xim"]:
            conv = (
                self.model.cosmo_model.kpc_comoving_per_arcmin(self.z_l + 1e-6)
                .to("Mpc / arcmin")
                .value
            )

            sep_out = sep_out / (conv * self.h)

        if self.components:
            return (
                sep_out,
                np.array(xi_list),
                np.array(xi_1h_list),
                np.array(xi_2h_list),
            )
        return sep_out, np.array(xi_list)
