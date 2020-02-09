'''
Computational experiments investigating behavior of the identity
signalling model over different parameter settings and other experimental
treatments.
'''
import matplotlib.pyplot as plt
import numpy as np

from .model import Model


def vary_covert_receiving_prob(covert_rec_probs=[0.05, 0.15, 0.25, 0.35, 0.45],
                               R=0.5, n_trials=10, n_iter=100,
                               **model_kwargs):

    # Initialize dictionary keyed by covert receiving probabilities;
    # values are 2D n_trials x n_iter+1 numpy arrays. Adding 1 to the number
    # of iterations in dimension bc we include 0th/initial value, set on
    # initialization of Model.
    results = {k: np.zeros((n_trials, n_iter + 1))
               for k in covert_rec_probs}

    for covert_rec_prob in covert_rec_probs:
        for trial_idx in range(n_trials):

            print(f"Covert receiving probability: {covert_rec_prob}\n"
                  f"Trial {trial_idx + 1} / 10")

            model = Model(prop_overt=R, prop_covert=covert_rec_prob,
                          **model_kwargs)
            model.run(n_iter)

            results[covert_rec_prob][trial_idx] = model.prop_covert_series

    return results


def analyze_covert_receiving_prob(results):

    # Calculate mean series for each covert receiving probability.
    mean_series = {
        covert_rec_prob: results_array.mean(axis=0)
        for covert_rec_prob, results_array in results.items()
    }

    # Plot them.
    for covert_rec_prob, series in mean_series.items():
        plt.plot(series, label=f"r/R = {covert_rec_prob / 0.5}")
