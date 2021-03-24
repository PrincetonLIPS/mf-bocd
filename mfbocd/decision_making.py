"""RL posterior form of information gain, MI(R, X).

NB: Computing MI(X, R) is distribution dependent; if such implementations
exist, they are in the model-specific files.
"""

import numpy as np
from   scipy.special import logsumexp

_SMALL = 1e-20


# -----------------------------------------------------------------------------

def rl_info_gain(t, model, log_message, hazard, j, hg_points, hg_weights):
    """Compute mutual information w.r.t. run length posterior.
    """
    lhs = prev_rl_entropy(log_message, hazard)
    rhs = expected_rl_entropy(t, model, log_message, hazard, j, hg_points,
                              hg_weights)
    if (lhs - rhs) < 0:
        # Numerical issue.
        assert(t == 1)
        return _SMALL
    return lhs - rhs


def prev_rl_entropy(log_message, hazard):
    log_growth_probs = log_message + np.log(1 - hazard)
    log_cp_prob      = logsumexp(log_message + np.log(hazard))
    log_rl_joint     = np.append(log_cp_prob, log_growth_probs)
    rl_post          = np.exp(log_rl_joint - logsumexp(log_rl_joint))
    assert(np.isclose(np.sum(rl_post), 1))
    return -1 * np.sum(rl_post * np.log(rl_post + _SMALL))


def expected_rl_entropy(t, model, log_message, hazard, j, points, weights):
    """Estimate the expected run length entropy.
    """

    def marg_pred(xs):
        pis = model.pred_prob(t, xs, j=j)
        return np.sum(pis * model._R[t-1, :t], axis=1)

    def rl_post(xs):
        log_pis = model.log_pred_prob(t, xs, j=j)
        log_joint = log_rl_joint_vectorized(log_pis, log_message, hazard)
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def exp_rl_ent(xs):
        # `xs` is an array; broadcast the values across `axis=1`. This
        # allows `scipy.stats.norm` to evaluate each column of `xs` using
        # the `t` hypothesis parameters.
        xs = np.repeat(xs[:, None], t, axis=1)
        rp = rl_post(xs)
        assert(xs.shape[0] == rp.shape[0])
        assert(np.allclose(rp.sum(axis=1), 1))
        entros = -1 * np.sum(rp * np.log(rp + _SMALL), axis=1)
        mp = marg_pred(xs)
        res = mp * entros
        return res

    return np.sum(weights * (exp_rl_ent(points) / np.exp(-points**2)))


def log_rl_joint_vectorized(log_pis, log_message, hazard):
    """Vectorized computation of the joint distribution. This makes quadrature
    fast, since we can evaluate all points as a vector.
    """
    log_growth_probs = log_pis + log_message + np.log(1 - hazard)
    log_cp_prob = logsumexp(log_pis + log_message + np.log(hazard), axis=1)
    log_cp_prob = log_cp_prob[:, None]
    return np.concatenate([log_cp_prob, log_growth_probs], axis=1)
