'''
Computational experiments investigating behavior of the identity
signalling model over different parameter settings and other experimental
treatments.
'''
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from functools import partial

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

    # Initialize process pool that will run trials in parallel.
    pool = mp.Pool(processes=4)
    for covert_rec_prob in covert_rec_probs:

        # Partial evaluation of _one_trial, providing necessary arguments.
        # XXX May be able to use args= and kws={} in apply_async.
        trial_func = partial(_one_trial, n_iter=n_iter, covert_rec_prob=covert_rec_prob,
                             R=R, **model_kwargs)

        pooled_trials = [
            pool.apply_async(trial_func)
            for _ in range(n_trials)
        ]

        for trial_idx, process_res in enumerate(pooled_trials):
            results[covert_rec_prob][trial_idx] = process_res.get()

    return results


def _one_trial(n_iter, covert_rec_prob, R, **model_kwargs):

    # Initialize model for trial.
    model = Model(prob_overt_receiving=R,
                  prob_covert_receiving=covert_rec_prob,
                  **model_kwargs)

    # Run model for desired number of iterations.
    model.run(n_iter)

    # Models have an attribute representing proportion of covert signalers.
    return model.prop_covert_series


def analyze_covert_receiving_prob(results):

    # Calculate mean series for each covert receiving probability.
    mean_series = {
        covert_rec_prob: results_array.mean(axis=0)
        for covert_rec_prob, results_array in results.items()
    }

    # Plot them.
    for covert_rec_prob, series in mean_series.items():
        plt.plot(series, label=f"r/R = {covert_rec_prob / 0.5}")
