import click
import numpy as np
import subprocess

from subprocess import PIPE

from id_signaling.experiment import run_experiments


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
            click.option('--num_traits', '-K', type=int, default=3),
            click.option('--similarity_threshold', '-S', type=float,
                         default=0.5),
            click.option('--learning_alpha', type=float, default=1.25)
       )


## RUNEXP ##
@basic_decorator()
def run(experiment, param_vals, homophily_vals, n_iter, n_trials,
        output_file, prob_overt_receiving, minority_trait_frac,
        initial_prop_covert, initial_prop_churlish, num_traits,
        similarity_threshold, learning_alpha):

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

    out_df = run_experiments(param_vals, homophily_vals, experiment, n_trials,
                             n_iter, prob_overt_receiving=prob_overt_receiving,
                             minority_trait_frac=minority_trait_frac,
                             initial_prop_covert=initial_prop_covert,
                             initial_prop_churlish=initial_prop_churlish,
                             K=num_traits,
                             similarity_threshold=similarity_threshold,
                             learning_alpha=learning_alpha
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
            similarity_threshold, learning_alpha,
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
    -K{num_traits} -S{similarity_threshold} \\
    --initial_prop_covert={initial_prop_covert} \\
    --initial_prop_churlish={initial_prop_churlish} \\
    --learning_alpha={learning_alpha}


printf "******************\\nFinished at `uptime`"
'''

    if dry_run:
        print(subscript)

    else:
        subprocess.run(['sbatch'], stdout=PIPE,
                       input=bytearray(subscript, 'utf-8'))
