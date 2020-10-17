import click
import concurrent
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess

from glob import glob
from subprocess import PIPE

from id_signaling.experiment import run_experiments
from id_signaling.figures import (
    heatmap, plot_correlation, plot_coevolution, invasion_heatmaps,
)


def composed(*decs):
    """Compose a series of decorators, which will be handy here."""
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


def basic_decorator():

    return composed(
            click.command(),
            click.argument('experiment', type=str),
            click.argument('param_vals', type=str),
            click.argument('homophily_vals', type=str),
            click.argument('n_iter', type=int),
            click.argument('n_trials', type=int),
            click.argument('output_file'),
            click.option('--prob_overt_receiving', '-R',
                         type=float, default=0.5),
            click.option('--minority_trait_frac', '-m'),
            click.option('--initial_prop_covert', type=float, default=0.5),
            click.option('--initial_prop_churlish', type=float, default=0.5),
            click.option('--num_traits', '-K', type=int, default=9),
            click.option('--similarity_threshold', '-S', type=float,
                         default=0.5),
            click.option('--learning_beta', type=float, default=10.0),
            click.option('--n_agents', '-N', type=int, default=100),
            click.option('--n_rounds', type=int, default=100),
            click.option('--two_dislike_penalty', type=float)
            # click.option('--two_dislike_penalty', type=float, default=0.25)
       )


## RUNEXP ##
@basic_decorator()
def run(experiment, param_vals, homophily_vals, n_iter, n_trials,
        output_file, prob_overt_receiving, minority_trait_frac,
        initial_prop_covert, initial_prop_churlish, num_traits,
        similarity_threshold, learning_beta, n_agents, n_rounds,
        two_dislike_penalty):

    # XXX Hack to deal with subexp throwing error I don't understand that
    # minority_trait_frac can't be None (the default) because it's not a
    # float. This doesn't seem to happen just with runexp.
    print(minority_trait_frac)
    if minority_trait_frac == 'None':
        minority_trait_frac = None
    if minority_trait_frac is not None:
        minority_trait_frac = float(minority_trait_frac)

    param_vals = np.arange(*[float(val) for val in param_vals.split(':')])
    homophily_vals = np.arange(*[float(val) for val in homophily_vals.split(':')])

    if two_dislike_penalty == 'None':
        two_dislike_penalty = None

    out_df = run_experiments(param_vals, homophily_vals, experiment, n_trials,
                             n_iter, prob_overt_receiving=prob_overt_receiving,
                             minority_trait_frac=minority_trait_frac,
                             initial_prop_covert=initial_prop_covert,
                             initial_prop_churlish=initial_prop_churlish,
                             K=num_traits,
                             similarity_threshold=similarity_threshold,
                             learning_beta=learning_beta, N=n_agents,
                             two_dislike_penalty=two_dislike_penalty
                             )

    out_df.to_csv(output_file, index=False)


## SUBEXP ##
@basic_decorator()
@click.option('--queue', '-q', type=str, default='fast.q')
@click.option('--ncpu', '-n', type=int, default=100)
@click.option('--wall_time', '-t', type=str, default='04:00:00')
@click.option('--dry_run', '-d', is_flag=True)
@click.option('--job_name', '-j', default=None)
def sub(
            experiment, param_vals, homophily_vals, n_iter, n_trials,
            output_file, prob_overt_receiving, minority_trait_frac,
            initial_prop_covert, initial_prop_churlish, num_traits,
            similarity_threshold, learning_beta, n_agents, n_rounds,
            two_dislike_penalty,  # END MODEL OPTS
            queue, ncpu, wall_time, dry_run, job_name  # SLURM OPTS
        ):
    """
    Submit experiment trials to the cluster using slurm .sub template.
    """

    if job_name is None:
        if minority_trait_frac is not None:
            job_name = f'minority_{minority_trait_frac}'
        else:
            job_name = experiment

    subscript = \
f'''#! /bin/bash
#SBATCH -p {queue}
#SBATCH -J {job_name}
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err
#SBATCH -n 1
#SBATCH -c {ncpu}
#SBATCH -t {wall_time}

printf "******************\\nStarting {job_name} at `uptime`\\n"

runexp {experiment} {param_vals} {homophily_vals} {n_iter} {n_trials} \\
    {output_file} -R{prob_overt_receiving} -m{minority_trait_frac} \\
    -K{num_traits} -S{similarity_threshold} -N{n_agents} \\
    --initial_prop_covert={initial_prop_covert} \\
    --initial_prop_churlish={initial_prop_churlish} \\
    --learning_beta={learning_beta} \\
    --n_rounds={n_rounds} \\
'''
    subscript_end = ''
    # Build end of submission script either with or without the two
    # disliking penalty.
    if two_dislike_penalty is None:

        subsciprt_end = \
'''

printf "******************\\nFinished at `uptime`"
'''
    else:

        subscript_end = \
'''
    --two_dislike_penalty={two_dislike_penalty}


printf "******************\\nFinished at `uptime`"
'''

    subscript += subscript_end

    if dry_run:
        print(subscript)

    else:
        subprocess.run(['sbatch'], stdout=PIPE,
                       input=bytearray(subscript, 'utf-8'))

## BEGIN ANALYSIS SCRIPT COMPONENTS ##
#
#
# Below I aggregate "part" files produced by each cluster nodes.
@click.group()
def run_analysis():
    pass


@run_analysis.command()
@click.option('--data_dir', default='data/basic',
              help='Location of data')
@click.option('--figure_dir', default='data/basic/Figures',
              help='Location to store figures')
def basic(data_dir, figure_dir):

    dirs = [os.path.join(data_dir, exp_type)
            for exp_type in ('disliking', 'receptivity')]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(_create_basic_invasion_full_csv, dirs)

    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)

    disliking = pd.read_csv(os.path.join(dirs[0], 'full.csv'))
    receptivity = pd.read_csv(os.path.join(dirs[1], 'full.csv'))

    print('Making prevalence heatmaps for four signaling/receiving '
          f'strategies saving to {figure_dir}')
    _make_basic_prevalence_heatmaps(disliking, receptivity, figure_dir)

    print('Making signaling-receiving correlation plots, '
          f'saving to {figure_dir}')
    plot_correlation(disliking)
    plt.savefig(os.path.join(figure_dir, 'basic_disliking_correlation.pdf'))

    plot_correlation(receptivity)
    plt.savefig(os.path.join(figure_dir, 'basic_receptivity_correlation.pdf'))

    print('Making correlation plots for supplement, '
          f'saving to {figure_dir}')
    _make_correlation_plots(disliking, receptivity, figure_dir)


@run_analysis.command()
@click.option('--data_dir', default='data/invasion',
              help='Location of data')
@click.option('--figure_dir', default='data/invasion/Figures',
              help='Location to store figures')
def invasion(data_dir, figure_dir):

    dirs = [os.path.join(data_dir, exp_type)
            for exp_type in ('disliking', 'receptivity')]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(_create_basic_invasion_full_csv, dirs)

    disliking = pd.read_csv(os.path.join(dirs[0], 'full.csv'))
    receptivity = pd.read_csv(os.path.join(dirs[1], 'full.csv'))

    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)

    _make_invasion_success_heatmaps(disliking, receptivity, figure_dir)


def _create_basic_invasion_full_csv(directory):

    # Create full.csv if they do not already exist.
    full_file = os.path.join(directory, 'full.csv')
    if not os.path.exists(full_file):
        print(f'Creating {full_file}')
        part_files = glob(os.path.join(directory, 'part*.csv'))
        os.system(f'head -n1 {part_files[0]} > {full_file}')
        os.system(f'tail -q -n+2 {" ".join(part_files)} >> {full_file}')
    else:
        print(f'Found existing {full_file}')


def _make_basic_prevalence_heatmaps(disliking, receptivity, figure_dir):

    heatmap(disliking, experiment='disliking')
    plt.title(f'Proportion covert signalers', size=14)
    plt.savefig(os.path.join(figure_dir, 'basic_disliking_signaling.pdf'))

    heatmap(disliking, experiment='disliking', strategy='receiving')
    plt.title(f'Proportion churlish receivers', size=14)
    plt.savefig(os.path.join(figure_dir, 'basic_disliking_receiving.pdf'))

    heatmap(receptivity, experiment='receptivity')
    plt.title(f'Proportion covert signalers', size=14)
    plt.savefig(os.path.join(figure_dir, 'basic_receptivity_signaling.pdf'))

    heatmap(receptivity, experiment='receptivity', strategy='receiving')
    plt.title(f'Proportion churlish receivers', size=14)
    plt.savefig(os.path.join(figure_dir, 'basic_receptivity_receiving.pdf'))


def _make_correlation_plots(disliking, receptivity, figure_dir):

    def make_path(f): return os.path.join(figure_dir, f)

    param_vals = [0.1, 0.5, 0.9]
    hvals = [0.1]
    plot_coevolution(disliking, 'disliking', param_vals, hvals,
                     savefig_path=make_path('disliking_evo_w=0p1.pdf'))

    param_vals = [0.1, 0.5, 0.9]
    hvals = [0.4]
    plot_coevolution(disliking, 'disliking', param_vals, hvals,
                     savefig_path=make_path('disliking_evo_w=0p4.pdf'))

    hvals = [0.1, 0.5, 0.9]
    param_vals = [0.1]
    plot_coevolution(disliking, 'disliking', param_vals, hvals,
                     savefig_path=make_path('disliking_evo_d=0p1.pdf'))

    hvals = [0.1, 0.5, 0.9]
    param_vals = [0.4]
    plot_coevolution(disliking, 'disliking', param_vals, hvals,
                     savefig_path=make_path('disliking_evo_d=0p4.pdf'))

    param_vals = [0.1, 0.5, 0.9]
    hvals = [0.1]
    plot_coevolution(receptivity, 'receptivity', param_vals, hvals,
                     savefig_path=make_path('receptivity_evo_w=0p1.pdf'))

    param_vals = [0.1, 0.5, 0.9]
    hvals = [0.4]
    plot_coevolution(receptivity, 'receptivity', param_vals, hvals,
                     savefig_path=make_path('receptivity_evo_w=0p4.pdf'))

    hvals = [0.1, 0.5, 0.9]
    param_vals = [0.1]
    plot_coevolution(receptivity, 'receptivity', param_vals, hvals,
                     savefig_path=make_path('receptivity_evo_r=0p1.pdf'))

    hvals = [0.1, 0.5, 0.9]
    plot_coevolution(receptivity, 'receptivity', param_vals, hvals,
                     savefig_path=make_path('receptivity_evo_r=0p4.pdf'))


def _make_invasion_success_heatmaps(disliking, receptivity, figure_dir):

    def make_path(f): return os.path.join(figure_dir, f)

    # Currently hardcoded since it works.
    invading_prev = 0.05

    import warnings; warnings.simplefilter('ignore')
    print(f'Making covert invasion heatmaps, saving to {figure_dir}')
    invasion_heatmaps(disliking, receptivity, invading='covert', timesteps=100,
                      save_path=make_path('covert_invades.pdf'),  # annot=True,
                      figsize=(10, 7),
                      invading_prev=invading_prev,
                      cbar_label_size=13)

    print(f'Making overt invasion heatmaps, saving to {figure_dir}')
    invasion_heatmaps(disliking, receptivity, invading='overt', timesteps=100,
                      save_path=make_path('overt_invades.pdf'),  # annot=True,
                      figsize=(10, 7),
                      invading_prev=invading_prev,
                      cbar_label_size=13)

    print(f'Making churlish invasion heatmaps, saving to {figure_dir}')
    invasion_heatmaps(disliking, receptivity, invading='churlish', timesteps=100,
                      save_path=make_path('churlish_invades.pdf'),  # annot=True,
                      figsize=(10, 7),
                      invading_prev=invading_prev,
                      cbar_label_size=13)

    print(f'Making generous invasion heatmaps, saving to {figure_dir}')
    invasion_heatmaps(disliking, receptivity, invading='generous', timesteps=100,
                      save_path=make_path('generous_invades.pdf'),  # annot=True,
                      figsize=(10, 7),
                      invading_prev=invading_prev,
                      cbar_label_size=13)
