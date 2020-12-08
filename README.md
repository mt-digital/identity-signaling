# The evolution of identity and covert signaling

This is the agent-based implementation of the model described in Smaldino & Turner (2020) "Covert signaling is an adaptive communication strategy in diverse populations".

The code is in several parts: the model, experiments using the model, analyses of experimental output, a command-line interface (CLI) for running experiments and analyses, and unit tests. Furthermore, in the root directory of the repo there is the `cl_scripts` directory that contains bash scripts that automate the submission of several jobs (experiments) at once. If your HPC cluster uses Slurm for scheduling these scripts may run as long as the necessary directories exist and have been modified as needed for your file system (please see the scripts in `cl_scripts` for more information on that).

Below are some notes on how to set up the code locally. For more information, please see function signatures, unit tests, and inline documentation, or [open an issue to ask a question](https://github.com/mt-digital/identity-signaling/issues). Thanks!


## Setup

Clone this repository and change into its directory. Start a virtualenv
and run the `pip install` command given below, which will enable
the command-line scripts.

All together,
```
git clone https://github.com/mt-digital/identity-signaling.git
cd identity-signaling
virtualenv venv
pip install --editable .
```

### Unit tests

Running unit tests is a good way to check all is well and to get started running or editing the code. To do this, as well as display how much of the model code base is covered by the unit tests, run the following command:

```sh
pytest --cov-report term --cov=id_signaling.model test/
```

It should print the following report, with some other surrounding information

```
test/test_model.py ............                                                                                [100%]

---------- coverage: platform darwin, python 3.7.6-final-0 -----------
Name                    Stmts   Miss  Cover
-------------------------------------------
id_signaling/model.py     232     11    95%
```

Each dot indicates one test that passed and the printout shows 95% of the code pass one unit test or
another. Of course the tests may be flawed, but these can be easily inspected for correctness in
relatioship to the formal model presented in the paper.

### Example of dry-run submit using the CLI 

Then submit a disliking/homophily experiment where disliking and homophily are
both varied over three values: 0.0, 0.25, and 0.5, using matlab-ish notation
with 40 iterations and 4 trials per parameter combination; save the file to
`disliking.csv`, and set some other parameters (run `subexp --help` for
details). To see the job submission script run `subexp` as a dry run using the `-d` flag
at the end of the command. We submit the script with all these options like so
```
subexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 40 4 disliking.csv -R0.5 -n400 -qstd.q -t01:00:00 -d
```

This prints out the job submission script

```
#! /bin/bash
#SBATCH -p std.q
#SBATCH -J disliking
#SBATCH -o disliking.out
#SBATCH -e disliking.err
#SBATCH -n 400
#SBATCH -t 01:00:00

printf "******************\nStarting disliking at `uptime`\n"

runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 4 40 \
    disliking.csv -R0.5 -mNone

printf "******************\nFinished at `uptime`"
```

This will be submitted to the cluster using the `squeue` command when the
`-d` flag is not passed to `subexp`. -d indicates "development".




## Experiments

Below is more on each experiment and code examples showing how to run them.

The output dataframes can be used in the `plot_evolution`, `heatmap`, and,
for the minority experiments, `minority_diff_heatmap` imported from
`id_signaling.figures`. There are a number of 
keyword arguments that must be specified to do 
plotting for each experiment. Examples are given below.


### Covert signaling receptivity

In this experiment we vary the receptivity of covert signaling. This determines
what fraction of the population receives covert signals. There is also a
receptivity of overt signals; the ratio of covert signal to overt signal
receptivities is the parameter used in the original 2018 paper.

To run `n_trials` of the covert signaling experiment where each trial is run
to `n_iter` for a given covert receptivity `r` and homophily `w`

```python
r = 0.05; w = 0.25
trials_df = trials_receptivity_homophily(r, w, n_iter=400, n_trials=100)
```

The resulting pandas dataframe has columns 

```python
['homophily', 'prop_churlish', 'prop_covert', 'receptivity', 'timestep',
       'trial_idx']
```

Use it with plotting functions like so

```python
plot_evolution(trials_df, experiment='receptivity', figsize=(7, 4.5))
heatmap(trials_df, experiment='receptivity')
```


## Disliking penalty

Here we vary the disliking penalty, which we set equal to the double-dislike
penalty. 

```python
from id_signaling.experiment import trials_dislikepen_homophily
d = 0.05; w = 0.25
trials_df = trials_dislikepen_homophily(d, w, n_iter=400, n_trials=100)
```

Output has columns

```python
['dislike', 'homophily', 'prop_churlish', 'prop_covert',
       'timestep', 'trial_idx']
```

In the notebooks I'm still using older .csv's that had `disliking_penalty`
instead of `dislike` for a column name, so there is the line

```python
df_r = df.rename(columns={'dislike_penalty': 'disliking'})
```

With this change for old data, or without for new data generated as shown
above, one then can run

```python
plot_evolution(trials_df, experiment='disliking')
heatmap(df_r, experiment='disliking')
```


## Minority experiment

For this experiment we have only looked at how disliking influence the 
differential evolution of covert signaling and churlish receiving in 
minority and majority populations. We can also use
`trials_receptivity_homophily` in place of `trials_dislikepen_homophily`
below. 

```python
trials_df = trials_dislikepen_homophily(exp_param_val, homophily, minority_trait_frac,
          experiment=experiment, n_trials=n_trials,
          n_iter=n_iter, R=R)
```

Now using the correct flags, `plot_evolution` and `heatmap` can be similarly
called on `trials_df`. In addition to these, the difference between majority
and minority covert signaling and churlish receiving in first and second 
lines below, respectively.

```python
minority_diff_heatmap(
    trials_df, savefig_path='reports/Figures/covert_signalers_diff.pdf'
)
minority_diff_heatmap(
    trials_df, strategy='receiving', vmin=-0.2, vmax=0.3, 
    savefig_path='reports/Figures/churlish_receivers_diff.pdf'
)
```


### Setting M based on K

For our experiments we are only setting K odd, and setting `M=K - (K+1)/2`. 
This occurs when a in the Model class constructor on initialization:

```python
class Model:
    def __init__(self, ...):
        ...
        if n_minmaj_traits is None:
            n_minmaj_traits = K - ((K+1) // 2)
        self.n_minmaj_traits = n_minmaj_traits
```

Doing this makes it so majority/minority agents only need to have one additional
trait in common with their majority/minority cohort in order to be similar
to them. In order for majority/minority agents to be similar to their outgroup,
they must have all `K - (K-1/2)` non-assigned traits in common. The chance of
this happening is `1/2^((K-1)/2)`. 


## Cluster scripts

See `run_disliking_homophily.py`, `run_receptivity_homophily.py`, and
`run_minority_experiment.py` for the scripts currently running on the
cluster. These need to be incorporated into a single CLIck API.

I also plan to put a job script creator/submitter in the CLI as well. This would
create a job script on the fly, submit the job with a certain parameter set
given by a command like `--homophily=0.0:1.01:0.025`, which would set 
five homophily parameters, 0, 0.25, 0.5, 0.75, 1.0. These values are used in
a call to `np.arange` which is why the max is set to 1.01 to pick up the 1.0
value.


### Create cluster directory structure

Eventually I want a script that creates the required directory structure, but
it will take some work to compile them all together. For now, I have to 
look at the different `sub_*.sh` files that make the proper series of 
calls to the Python CLI that run the desired experiments, and based on the
file names used create the directory structure that will accommodate 
the file names. Because the file name is used by the cluster to write .out
and .err files to, if an error has been made the cluster script will fail
immediately. 

Example: see `sub_tolerance_diversity_experiments.sh` which requires a directory
structure created by the following commands:

```sh
mkdir -p output_data/tolerance_diversity/0.{1..9}/"K="{10,15,20}/
mkdir -p output_data/tolerance_diversity/1.0/"K="{10,15,20}/
```

### Scripts to submit multiple versions

To spread a simulation across multiple runs, I wrote a number of scripts to 
submit a number of jobs to the cluster that will run all parameter values and
the right number of trials per parameter value. For example, here are the
main contents of `sub_minority_K=9;M=4.sh`:

```sh
for i in {1..10}; do
    fname=output_data/minority_K9M4/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
        --n_minmaj_traits=4 -K9 -m0.10 \
        -qfast.q -j$fname -n24 -t"04:00:00"
        # -qstd.q -j$fname -n20 -t"04:00:00" --n_minmaj_traits=4 -K10
done
```

This writes all data to randomly-named files to the directory
`output_data/minority_K9M4`. To create a single CSV, I have been running the
following commands, using an example randomly-generated file name. The first
line grabs the header and the second line takes all lines but the header from
each file and puts the data in full.csv.

```
$ head -n1 part-a917cc22-14e9-4716-8b66-8a610028ed3f.csv > full.csv
$ tail -q -n+2 part-*.csv >> full.csv
```

### Fixing failures of experiment parts

In the process of running these different parameters and creating different
part files, there may be problems. For instance, due to cluster issues
(all clusters have them sometimes!), I had some trials dropped from the
Minority/Tolerance experiments. Specifically, 7 different parameter settings
out of the 80 tested (`K=3,9`, `M=1,4`, `S=0.1, 0.2, ..., 1.0`, and 
`rho_minor=0.1, 0.2`) failed. The seven are (3, 1, .3, 0.1), (3, 1, .4, 0.1), 
(3, 1, .5, 0.1), (3, 1, .3, 0.2), (3, 1, .5, 0.2), (3, 1, .7, 0.2), 
and (9, 4, .4, 0.2).

Here is how one could modify the lists of parameter values to only
contain the missing parameters. Note that in all minority experiments we are
setting `M=K - (K+1)/2` (see above), so we do not inlcude explicit settings for `M`.
Here is a script one could run to get only those missing parameter combinations:

```bash
# Submit missing trials for K=3, rho_minor=0.1.
for similarity_threshold in 0.3 0.4 0.5; do
    K=3
    minority_trait_frac=0.1
    fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
        -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
        -K$K -S$similarity_threshold 
done

# Submit missing trials for K=3, rho_minor=0.2.
for similarity_threshold in 0.3 0.5 0.7; do
    K=3
    minority_trait_frac=0.2
    fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
        -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
        -K$K -S$similarity_threshold 
done

# Only one failed for K=9 across both similarity thresholds; no need to loop.
K=9
similarity_threshold=0.4
minority_trait_frac=0.2

fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
    -K$K -S$similarity_threshold 
```
