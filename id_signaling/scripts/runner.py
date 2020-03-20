import click
import numpy as np

from id_signaling.experiment import run_experiments


# @click.group()
# @click.pass_context
# def run(ctx):
#     "Run a series of given experiment trials"
#     ctx.obj = dict()

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
        click.option('--minority_trait_frac', '-m',
                     type=float, default=None),
   )

## RUNEXP ##
@basic_decorator()
def run(experiment, param_vals, homophily_vals, n_trials, n_iter,
        output_file, prob_overt_receiving, minority_trait_frac):

    param_vals = np.arange(*[float(val) for val in param_vals.split(':')])
    homophily_vals = np.arange(*[float(val) for val in homophily_vals.split(':')])
    # XXX This is where to go to edit parallelization. See note in run_experiments XXX
    out_df = run_experiments(param_vals, homophily_vals, experiment, n_trials,
                             n_iter, prob_overt_receiving=prob_overt_receiving,
                             minority_trait_frac=minority_trait_frac)

    out_df.to_csv(output_file, index=False)

## SUBEXP ##
@basic_decorator()
@click.option('--queue', '-q', type=str, default='fast.q')
@click.option('--ncpu', '-n', type=int, default=100)
@click.option('--wall_time', '-t', type=str, default='04:00:00')
@click.option('--dry_run', '-d', is_flag=True)
def sub(
        experiment, param_vals, homophily_vals, n_trials, n_iter,
        output_file, overt_receptivity, minority_trait_frac,
        queue, ncpu, wall_time, dry_run
    ):
    """
    Submit experiment trials to the cluster using slurm .sub template.
    """

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
#SBATCH -n {ncpu}
#SBATCH -t {wall_time}

printf "******************\\nStarting {job_name} at `uptime`\\n"

runexp {experiment} {param_vals} {homophily_vals} {n_trials} {n_iter} \\
    {output_file} -R{overt_receptivity} -m{minority_trait_frac}

printf "******************\\nFinished at `uptime`"
'''

    if dry_run:
        print(subscript)

    else:
        pass  # TODO submit the subscript using squeue
