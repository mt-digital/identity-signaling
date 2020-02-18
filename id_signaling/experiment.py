'''
Computational experiments investigating behavior of the identity
signalling model over different parameter settings and other experimental
treatments.
'''
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import time

from functools import partial
from time import sleep

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
    # pooled_trials = list()

    # Need to pick random seeds at random
    seeds = np.random.randint(2**32 - 1,
                              size=(len(covert_rec_probs), n_trials))

    for prob_idx, covert_rec_prob in enumerate(covert_rec_probs):

        with mp.Pool(processes=4) as pool:

            trial_func = partial(_one_trial, n_iter=n_iter,
                                 covert_rec_prob=covert_rec_prob,
                                 R=R, **model_kwargs)

            these_seeds = seeds[prob_idx]

            results[covert_rec_prob] = np.array(
                list(pool.map(trial_func, these_seeds))
            )

    return results


def trials_receptivity_homophily(receptivity, homophily, n_trials=10,
                                 n_iter=100, R=0.5):

    results_covert = np.zeros((n_trials, n_iter + 1))
    results_churlish = np.zeros((n_trials, n_iter + 1))
    seeds = np.random.randint(2**32 - 1, size=(n_trials,))

    with mp.Pool(processes=mp.cpu_count()) as pool:

        trial_func = partial(_one_trial, n_iter=n_iter,
                             covert_rec_prob=receptivity,
                             R=R, homophily=homophily)
        results = list(pool.map(trial_func, seeds))
        results_covert = np.array([el[0] for el in results])
        results_churlish = np.array([el[1] for el in results])

    return pd.DataFrame(
        {
            "timestep": list(range(n_iter + 1)) * n_trials,

            "trial_idx": [idx
                          for _ in range(n_iter + 1)
                          for idx in range(n_trials)],

            "homophily": [homophily] * n_trials * (n_iter + 1),

            "receptivity": [receptivity] * n_trials * (n_iter + 1),

            "prop_covert": results_covert.flatten(),

            "prop_churlish": results_churlish.flatten()
        }
    )


def trials_dislikepen_homophily(dislike_penalty, homophily, n_trials=10,
                                n_iter=100, R=0.5):

    results_covert = np.zeros((n_trials, n_iter + 1))
    results_churlish = np.zeros((n_trials, n_iter + 1))
    seeds = np.random.randint(2**32 - 1, size=(n_trials,))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        trial_func = partial(_one_trial, n_iter=n_iter,
                             covert_rec_prob=0.25,
                             one_dislike_penalty=dislike_penalty,
                             two_dislike_penalty=dislike_penalty,
                             R=R, homophily=homophily)
        # results = np.array(
        #     list(pool.map(trial_func, seeds))
        # )
        results = list(pool.map(trial_func, seeds))
        results_covert = np.array([el[0] for el in results])
        results_churlish = np.array([el[1] for el in results])


    return pd.DataFrame(
        {
            "timestep": list(range(n_iter + 1)) * n_trials,

            "trial_idx": [idx
                          for _ in range(n_iter + 1)
                          for idx in range(n_trials)],

            "dislike_penalty": [dislike_penalty] * n_trials * (n_iter + 1),

            "homophily": [homophily] * n_trials * (n_iter + 1),

            "prop_covert": results_covert.flatten(),

            "prop_churlish": results_churlish.flatten()
        }
    )


def _one_trial(seed, n_iter, covert_rec_prob, R, **model_kwargs):

    with open('log.txt', 'a+') as f:
        f.write(f'running trial with random seed {seed}\n')

    # Initialize model for trial.
    model = Model(prob_overt_receiving=R,
                  prob_covert_receiving=covert_rec_prob,
                  random_seed=seed,
                  **model_kwargs)

    # Run model for desired number of iterations.
    model.run(n_iter)

    # Models have an attribute representing proportion of covert signalers.
    return (model.prop_covert_series, model.prop_churlish_series)


def analyze_covert_receiving_prob(results):

    # Calculate mean series for each covert receiving probability.
    mean_series = {
        covert_rec_prob: results_array.mean(axis=0)
        for covert_rec_prob, results_array in results.items()
    }

    # Plot them.
    for covert_rec_prob, series in mean_series.items():
        plt.plot(series, label=f"r/R = {covert_rec_prob / 0.5}")
