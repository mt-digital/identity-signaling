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


def trials_minority(exp_param, homophily, minority_trait_frac=0.1,
                    experiment='disliking',
                    n_trials=10, n_iter=100, R=0.5):

    ###  XXX XXX XXX  ###
    # Start here in AM. #
    #####################

    results_covert = np.zeros((n_trials, n_iter + 1))
    results_churlish = np.zeros((n_trials, n_iter + 1))
    seeds = np.random.randint(2**32 - 1, size=(n_trials,))

    with mp.Pool(processes=mp.cpu_count()) as pool:

        if experiment == 'receptivity':
            trial_func = partial(_one_minority_trial, n_iter=n_iter,
                                 covert_rec_prob=exp_param,
                                 minority_trait_frac=minority_trait_frac,
                                 R=R, homophily=homophily)

        elif experiment == 'disliking':
            trial_func = partial(_one_minority_trial, n_iter=n_iter,
                                 covert_rec_prob=0.25,
                                 one_dislike_penalty=exp_param,
                                 two_dislike_penalty=exp_param,
                                 minority_trait_frac=minority_trait_frac,
                                 R=R, homophily=homophily)
        else:
            raise RuntimeError(f'{experiment} not recognized')

        results = list(pool.map(trial_func, seeds))

        r0 = results[0]
        results_dict = {k: [] for k in r0.keys()}
        for result in results:
            for k, v in result.items():
                results_dict[k].append(v)

        ret_series_data = {k: np.array(v).flatten()
                           for k, v in results_dict.items()}
            # results_dict.update({
            #     k: results_dict[k].append(v) for k, v in result.items()
            # })

    ret_dict = {
        'timestep': list(range(n_iter + 1)) * n_trials,

        'trial_idx': [idx
                      for _ in range(n_iter + 1)
                      for idx in range(n_trials)],

        'homophily': [homophily] * n_trials * (n_iter + 1),

        # XXX this suggests changing column to 'disliking' for that
        # experiment.
        experiment: [exp_param] * n_trials * (n_iter + 1)
        }

    ret_dict.update(ret_series_data)

    ret = pd.DataFrame(ret_dict)

    return ret

            # "prop_covert": results_covert.flatten(),

            # "prop_churlish": results_churlish.flatten(),

            # "prop_covert_minority": results_covert_minority.flatten(),
            # "prop_churlish_minority": results_churlish_minority.flatten(),

            # "prop_covert_majority": results_covert_majority.flatten(),
            # "prop_churlish_majority": results_churlish_majority.flatten()
        # }


def run_experiments(exp_param_vals, homophily_vals, experiment='receptivity',
                    n_trials=4, n_iter=50, R=0.5, minority_trait_frac=None):

    if minority_trait_frac is not None:
        func = trials_minority
    elif experiment == 'receptivity':
        func = trials_receptivity_homophily
    elif experiment == 'disliking':
        func = trials_dislikepen_homophily
    else:
        raise RuntimeError(f'{experiment} not recognized')

    # Run experiment trials for each parameter setting and append to returned
    # dataframe with all trial data.
    df_full = None

    for exp_param_val in exp_param_vals:
        for homophily in homophily_vals:

            if minority_trait_frac is not None:
                df = func(exp_param_val, homophily, minority_trait_frac,
                          experiment=experiment, n_trials=n_trials,
                          n_iter=n_iter, R=R)
            else:
                df = func(val, val, n_iter=n_iter, n_trials=n_trials, R=R)

            if df_full is None:
                df_full = df
            else:
                df_full = df_full.append(df)


    return df_full

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

            "dislike": [dislike_penalty] * n_trials * (n_iter + 1),

            "homophily": [homophily] * n_trials * (n_iter + 1),

            "prop_covert": results_covert.flatten(),

            "prop_churlish": results_churlish.flatten()
        }
    )


def _one_trial(seed, n_iter, covert_rec_prob, R, **model_kwargs):

    print(f'running trial with random seed {seed}')

    # Initialize model for trial.
    model = Model(prob_overt_receiving=R,
                  prob_covert_receiving=covert_rec_prob,
                  random_seed=seed,
                  **model_kwargs)

    # Run model for desired number of iterations.
    model.run(n_iter)

    # Models have an attribute representing proportion of covert signalers.
    return (model.prop_covert_series, model.prop_churlish_series)


def _one_minority_trial(seed, n_iter, minority_trait_frac,
                        covert_rec_prob, R, **model_kwargs):

    print(f'running trial with random seed {seed}')

    # Initialize model for trial.
    model = Model(prob_overt_receiving=R,
                  prob_covert_receiving=covert_rec_prob,
                  minority_trait_frac=minority_trait_frac,
                  random_seed=seed,
                  **model_kwargs)

    # Run model for desired number of iterations.
    model.run(n_iter)

    # Models have an attribute representing proportion of covert signalers.
    return {
        'prop_covert': model.prop_covert_series,
        'prop_churlish': model.prop_churlish_series,
        'prop_covert_minority': model.prop_covert_series_minority,
        'prop_churlish_minority': model.prop_churlish_series_minority,
        'prop_covert_majority': model.prop_covert_series_majority,
        'prop_churlish_majority': model.prop_churlish_series_majority,
    }

def analyze_covert_receiving_prob(results):

    # Calculate mean series for each covert receiving probability.
    mean_series = {
        covert_rec_prob: results_array.mean(axis=0)
        for covert_rec_prob, results_array in results.items()
    }

    # Plot them.
    for covert_rec_prob, series in mean_series.items():
        plt.plot(series, label=f"r/R = {covert_rec_prob / 0.5}")
