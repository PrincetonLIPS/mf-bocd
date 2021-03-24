import numpy as np
from   scipy.special import logsumexp

_SMALL = 1e-20


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Perform Bayesian online changepoint detection.
    """
    data, T, dim = _check_input(data)

    log_message = np.array([1])

    for t in range(1, T + 1):
        # 2. Observe new datum.
        x = data[t - 1]

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        # 5. Calculate changepoint probabilities.
        new_log_joint = log_rl_joint(log_pis, log_message, hazard)

        # 6. Calculate evidence
        # 7. Determine run length distribution.
        model.update_rl_posterior(t, new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Setup message passing.
        log_message = new_log_joint

        # Make model predictions.
        if t < T:
            model.predict(t)

    return model


# -----------------------------------------------------------------------------

def mf_bocd(data, model, hazard, choose_fid, return_data=True):
    """Perform active multi-fidelity Bayesian online changepoint detection.
    """
    data, J, T, dim = _check_input_mf(data)
    if dim == 1:
        data_chosen = np.empty(T)
    else:
        data_chosen = np.empty((T, dim))

    log_message = np.array([1])

    # MI calculations.
    choices = np.zeros(T)
    igs = np.zeros((J, T))

    for t in range(1, T + 1):

        # 1. Choose fidelity.
        igs[:, t-1] = model.compute_ig(t, log_message, hazard)
        j = choose_fid(igs[:, t-1])
        choices[t-1] = j

        # 2. Observe new datum.
        x = data[j, t-1]

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x, j=j)

        # 4. Calculate growth probabilities.
        # 5. Calculate changepoint probabilities.
        new_log_joint = log_rl_joint(log_pis, log_message, hazard)

        # 6. Calculate evidence
        # 7. Determine run length distribution.
        model.update_rl_posterior(t, new_log_joint)
        model.update_params(t, x, j=j)

        data_chosen[t - 1] = x
        log_message = new_log_joint

        if t < T:
            model.predict(t, j=j)

    if return_data:
        return choices, igs, data_chosen
    else:
        return choices, igs


# -----------------------------------------------------------------------------
# Utility functions.
# -----------------------------------------------------------------------------

def log_rl_joint(log_pis, log_message, hazard):
    """Compute joint distribution p(r_{t} | x_{1:t}, s_{1:t}).
    """
    log_growth_probs = log_pis + log_message + np.log(1 - hazard)
    log_cp_prob      = logsumexp(log_pis + log_message + np.log(hazard))
    return np.append(log_cp_prob, log_growth_probs)


def _check_input(data):
    """Check input for multi-fidelity data. Return data, number of samples T,
    and data dimension dim.
    """
    T    = len(data)
    data = np.array(data)
    dim  = data[0].size if not np.isscalar(data[0]) else 1
    return data, T, dim


def _check_input_mf(data):
    """Check input for multi-fidelity data. Return data, number of fidelities
    J, number of samples T, and data dimension dim.
    """
    data = np.array(data)
    J, T = data.shape[:2]
    dim  = data[0, 0].size if not np.isscalar(data[0, 0]) else 1
    return data, J, T, dim


def _init_prediction_vars(T, dim):
    """Initialize variables for prediction.
    """
    if dim > 1:
        pmean = np.zeros((T, dim))
        pvar = np.zeros((T, dim))
    else:
        pmean = np.zeros(T)
        pvar = np.zeros(T)
    return pmean, pvar
