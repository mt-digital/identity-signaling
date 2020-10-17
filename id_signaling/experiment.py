'''
Computational experiments investigating behavior of the identity
signalling model over different parameter settings and other experimental
treatments.
'''
import click
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import time

from functools import partial
from time import sleep

from .model import Model


def run_experiments(exp_param_vals, homophily_vals, experiment='receptivity',
                    n_trials=4, n_iter=50, prob_overt_receiving=0.5,
                    minority_trait_frac=None, **extra_model_kwargs):
    '''
    extra_model_kwargs could be used for sensitivity analyses to test other
    parameters.
    '''
    # Run experiment trials for each parameter setting and append to returned
    # dataframe with all trial data.
    df_full = None

    # Create array of input data over which to map _one_trial.
    n_seeds = n_trials * len(exp_param_vals) * len(homophily_vals)
    seeds = np.random.randint(2**32 - 1, size=(n_seeds,))

    params_and_seeds = []
    seed_index = 0
    for param in exp_param_vals:
        for homophily in homophily_vals:
            for _ in range(n_trials):
                params_and_seeds.append((seeds[seed_index], param, homophily))
                seed_index += 1

    one_trial = partial(_one_trial, experiment=experiment, n_iter=n_iter,
                         prob_overt_receiving=prob_overt_receiving,
                         minority_trait_frac=minority_trait_frac,
                         **extra_model_kwargs)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(pool.map(one_trial, params_and_seeds))

    return pd.concat(results)


def _one_trial(trial_tup, experiment, n_iter, **model_kwargs):

    seed = trial_tup[0]
    exp_param = trial_tup[1]
    homophily = trial_tup[2]

    # print(f'running trial with random seed {seed}')
    click.echo(f'running trial with random seed {seed}')

    # Initialize model for trial.
    if experiment == 'disliking':
        # XXX Not sure what the point of "experiment" and "model" kwargs are...
        experiment_kwargs = dict(
            one_dislike_penalty=exp_param
        )
        # Set the model kwarg to be the experimental parameter, the
        # one-agent disliking penalty, d.
        # else:
        #     experiment_kwargs = dict(
        #         one_dislike_penalty=exp_param
        #     )

    elif experiment == 'receptivity':
        experiment_kwargs = dict(prob_covert_receiving=exp_param)

    if model_kwargs['two_dislike_penalty'] is None:  # not in model_kwargs:
        model_kwargs['two_dislike_penalty'] = exp_param

    model = Model(homophily=homophily, **experiment_kwargs, **model_kwargs)

    # Run model for desired number of iterations.
    model.run(n_iter)

    one_dislike_penalty = model.one_dislike_penalty
    two_dislike_penalty = model.two_dislike_penalty

    # Models have an attribute representing proportion of
    # covert and churlish signalers.  If minority trait frac is given we need
    # the proportion of covert and overt in the majority and minority.
    # A smarter way prob exists, but this works.
    n_tstep = n_iter + 1
    ret = dict(
        timestep=np.arange(0, n_tstep, dtype=int),
        trial_idx=[seed]*n_tstep,
        initial_prop_covert=[model.initial_prop_covert]*n_tstep,
        initial_prop_churlish=[model.initial_prop_churlish]*n_tstep,
        prop_covert=model.prop_covert_series,
        prop_churlish=model.prop_churlish_series,
        homophily=[homophily] * n_tstep,
        K=[model.K] * n_tstep,
        S=[model.similarity_threshold] * n_tstep,
        M=[model.n_minmaj_traits] * n_tstep,
        disliking=model.one_dislike_penalty,
        two_dislike_penalty=model.two_dislike_penalty
    )

    minority_trait_frac = model_kwargs['minority_trait_frac']
    if minority_trait_frac is not None:
        ret.update({
            'prop_covert_minority': model.prop_covert_series_minority,
            'prop_churlish_minority': model.prop_churlish_series_minority,
            'prop_covert_majority': model.prop_covert_series_majority,
            'prop_churlish_majority': model.prop_churlish_series_majority,
            'homophily': [homophily] * n_tstep,
            'minority_trait_frac': [minority_trait_frac] * n_tstep,
        })

    if experiment == 'disliking':
        ret.update(dict(
            disliking=[exp_param] * n_tstep,
            two_dislike_penalty=[model.two_dislike_penalty] * n_tstep
        ))

    if experiment == 'receptivity':
        ret.update(dict(receptivity=[exp_param]*n_tstep))

    return pd.DataFrame(ret)


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
