import numpy as np
from   scipy.special import logsumexp


# -----------------------------------------------------------------------------

class BaseModel:

    def __init__(self, n_samples):
        self.T = n_samples
        # When we exponentiate R, exp(-inf) --> 0.
        self._log_R = np.ones((self.T + 1, self.T + 1)) * -np.inf
        self._log_R[0, 0] = 1
        self._pmean = np.zeros(self.T)
        self._pvar = np.zeros(self.T)

    def update_rl_posterior(self, t, new_log_joint):
        """Update run length posterior distribution.
        """
        self._log_R[t, :t+1] = new_log_joint
        self._log_R[t, :t+1] -= logsumexp(new_log_joint)

    def predict(self, t, **kwargs):
        """Make time t prediction, i.e. predict data at time t+1.
        """
        means, vars_ = self._get_posterior_params(t, j=kwargs.get('j', None))
        rl_post = self._R[t-1, :t]
        # Index is t-1 because we make no predictions at time t=0.
        self._pmean[t-1] = np.sum(means * rl_post)
        self._pvar[t-1]  = np.sum(vars_ * rl_post)

    def _get_posterior_params(self, t, **kwargs):
        """Compute model's predictive mean and variance.
        """
        raise NotImplementedError

    def pred_prob(self, t, x, **kwargs):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        raise NotImplementedError

    def log_pred_prob(self, t, x, **kwargs):
        raise NotImplementedError

    def update_params(self, t, x, **kwargs):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        raise NotImplementedError

    @property
    def _R(self):
        """Return run length posterior.
        """
        return np.exp(self._log_R)


# -----------------------------------------------------------------------------

class MfBaseModel(BaseModel):

    def __init__(self, n_samples, zetas):
        super().__init__(n_samples=n_samples)
        assert (np.all(0 <= np.array(zetas)))
        self.J = len(zetas)
        self.zetas = zetas

    def compute_ig(self, t, log_message, hazard):
        """Compute information gain.
        """
        raise NotImplementedError
