from mfbocd.models.base import (BaseModel,
                                MfBaseModel)
import numpy as np
from   scipy.special import gammaln

_SMALL = 1e-20


# -----------------------------------------------------------------------------

def log_beta_fun(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


# -----------------------------------------------------------------------------

class Bernoulli(BaseModel):

    def __init__(self, n_samples, alpha0, beta0):
        """Initialize model:

        theta ~ beta(alpha0, beta0)
        x     ~ Bernoulli(theta)
        """
        super().__init__(n_samples=n_samples)
        assert(np.isscalar(alpha0))
        assert(np.isscalar(beta0))
        self.alpha0       = alpha0
        self.beta0        = beta0
        self.alpha_params = np.array([self.alpha0])
        self.beta_params  = np.array([self.beta0])

    def _get_posterior_params(self, t, **kwargs):
        # For the posterior predictive variance, see
        # http://www.markirwin.net/stat220/Lecture/Lecture4.pdf
        a = self.alpha_params[:t]
        b = self.beta_params[:t]
        means = a / (a + b)
        vars_ = (a * b) / ((a + b)**2 * (a + b + 1))
        return means, vars_

    def pred_prob(self, t, x, **kwargs):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        return np.exp(self.log_pred_prob(t, x))

    def log_pred_prob(self, t, x, **kwargs):
        a = self.alpha_params[:t]
        b = self.beta_params[:t]
        numer = x * np.log(a + _SMALL) + (1 - x) * np.log(b + _SMALL)
        denom = np.log(a + b + _SMALL)
        return numer - denom

    def update_params(self, t, x, **kwargs):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        new_alpha_params  = self.alpha_params + x
        self.alpha_params = np.append(self.alpha0, new_alpha_params)
        new_beta_params   = self.beta_params + 1 - x
        self.beta_params  = np.append(self.beta0, new_beta_params)


# -----------------------------------------------------------------------------

class MfBernoulli(MfBaseModel):

    def __init__(self, n_samples, zetas, alpha0, beta0):
        """Initialize model:

        theta ~ beta(alpha0, beta0)
        x     ~ Bernoulli(theta)
        """
        super().__init__(n_samples=n_samples,
                         zetas=zetas)
        assert (0 <= alpha0)
        assert (0 <= beta0)

        self.alpha0       = alpha0
        self.beta0        = beta0
        self.alpha_params = np.array([self.alpha0])
        self.beta_params  = np.array([self.beta0])

    def _get_posterior_params(self, t, **kwargs):
        a     = self.alpha_params[:t]
        b     = self.beta_params[:t]
        means = a / (a + b)
        vars_ = (a * b) / ((a + b)**2 * (a + b + 1))
        return means, vars_

    def pred_prob(self, t, x, **kwargs):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        j = kwargs.get('j')
        return np.exp(self.log_pred_prob(t, x, j=j))

    def log_pred_prob(self, t, x, **kwargs):
        a = self.alpha_params[:t]
        b = self.beta_params[:t]
        z = self.zetas[kwargs.get('j')]
        numer = log_beta_fun((z*x) + a, (z*(1-x)) + b)
        denom = log_beta_fun(a, b)
        return numer - denom

    def update_params(self, t, x, **kwargs):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        zeta              = self.zetas[kwargs.get('j')]
        new_alpha_params  = self.alpha_params + (zeta * x)
        self.alpha_params = np.append(self.alpha0, new_alpha_params)
        new_beta_params   = self.beta_params + (zeta * (1 - x))
        self.beta_params  = np.append(self.beta0, new_beta_params)

    def compute_ig(self, t, log_message, hazard):
        igs = np.empty(self.J)
        for j in range(self.J):
            igs[j] = pred_info_gain_mf(t, self, j)
        return igs


# -----------------------------------------------------------------------------
# Information gain.
# -----------------------------------------------------------------------------

def pred_info_gain_mf(t, model, j):
    lhs = prev_predictive_entropy(t, model, j)
    rhs = exp_predictive_entropy(t, model, j)
    igs = lhs - rhs
    if igs < 0:
        assert(t == 1)
        return _SMALL
    return igs


def prev_predictive_entropy(t, model, j):

    def marg_pred(x):
        pis = model.pred_prob(t, x, j=j)
        return np.sum(pis * model._R[t-1, :t])

    def entro_marg_pred(x):
        mp = marg_pred(x)
        return -1 * mp * np.log(mp + _SMALL)

    lhs = 0
    for x in [0, 1]:
        lhs += entro_marg_pred(x)
    return lhs


def exp_predictive_entropy(t, model, j):
    entros = np.zeros(t)
    for x in [0, 1]:
        pp = model.pred_prob(t, x, j=j)
        entros += -1 * pp * np.log(pp + _SMALL)
    rhs = np.sum(model._R[t-1, :t] * entros)
    return rhs
