'''
Computational experiments investigating behavior of the identity
signalling model over different parameter settings and other experimental
treatments.
'''
import numpy as np

from model import Model


def vary_homophily(values=[0, 0.25, 0.5], n_trials=5):

    # Set up results dictionary with final proportion covert and churlish.
    res = {
        k: np.zeros((len(values), n_trials))
        for k in ('prop_covert', 'prop_churlish')
    }

    # Run a model for each homophily value, n_trials trials for each value.
    for value_idx, value in enumerate(values):
        for trial_idx in range(n_trials):
            res['prop_covert'][value_idx, trial_idx] = \
                _proportion_covert(model)
            res['prop_churlish'][value_idx, trial_idx] = \
                _proportion_churlish(model)

    return res


#: Calculate proportion of covert agents in a Model instance.
def _proportion_covert(model):
    return (
        np.sum(
            [a.signaling_strategy == "Covert" for a in model.agents]
        ) / model.N
    )


#: Calculate proportion of churlish agents in a Model instance.
def _proportion_churlish(model):
    return (
        np.sum(
            [a.receiving_strategy == "Churlish" for a in model.agents]
        ) / model.N
    )
