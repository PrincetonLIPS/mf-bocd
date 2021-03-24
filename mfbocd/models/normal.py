import numpy as np
from   numpy.polynomial.hermite import hermgauss
from mfbocd.models.base import (BaseModel,
                                MfBaseModel)
from   scipy.stats import norm
from   mfbocd.decision_making import rl_info_gain

HG_POINTS, HG_WEIGHTS = hermgauss(deg=50)


# -----------------------------------------------------------------------------
# Univariate normal.
# -----------------------------------------------------------------------------

class Normal(BaseModel):

    def __init__(self, n_samples, mean0, var0, varx):
        """Initialize model:

        theta ~ N(mean0, var0)
        x     ~ N(theta, varx)
        """
        super().__init__(n_samples=n_samples)
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def _get_posterior_params(self, t, **kwargs):
        means = self.mean_params[:t]
        vars_ = self.var_params[:t]
        return means, vars_

    def pred_prob(self, t, x, **kwargs):
        return np.exp(self.log_pred_prob(t, x))

    def log_pred_prob(self, t, x, **kwargs):
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x, **kwargs):
        # See eq. 19 in (Murphy 2007).
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params = (self.mean_params * self.prec_params[:-1] + (
                    x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx


# -----------------------------------------------------------------------------

class MfNormal(MfBaseModel):

    def __init__(self, n_samples, zetas, mean0, var0, varx):
        super().__init__(n_samples=n_samples,
                         zetas=zetas)
        self.mean0       = mean0
        self.var0        = var0
        self.varx        = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def _get_posterior_params(self, t, **kwargs):
        zeta  = self.zetas[kwargs.get('j')]
        means = self.mean_params[:t]
        vars_ = self.var_params(zeta)[:t]
        return means, vars_

    def pred_prob(self, t, x, **kwargs):
        j = kwargs.get('j')
        return np.exp(self.log_pred_prob(t, x, j=j))

    def log_pred_prob(self, t, x, **kwargs):
        zeta       = self.zetas[kwargs.get('j')]
        post_means = self.mean_params[:t]
        post_vars  = self.var_params(zeta)[:t]
        post_stds  = np.sqrt(post_vars)
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x, **kwargs):
        zeta             = self.zetas[kwargs.get('j')]
        new_prec_params  = self.prec_params + (zeta/self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        new_mean_params  = (self.mean_params * self.prec_params[:-1]
                            + ((zeta * x) / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    def compute_ig(self, t, log_message, hazard):
        igs = np.empty(self.J)
        for j in range(self.J):
            igs[j] = rl_info_gain(t, self, log_message, hazard, j,
                                  HG_POINTS, HG_WEIGHTS)
        return igs

    def var_params(self, zeta):
        """Helper function for computing the posterior variance.
        """
        return (1./self.prec_params) + (self.varx/zeta)
