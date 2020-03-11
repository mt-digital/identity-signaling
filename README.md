# The evolution of identity and covert signaling

This model is the agent-based implementation of Smaldino, Flamson, \& 
McElreath (2018), ``The Evolution of Covert Signaling''. 

So far we have developed three experiments: _covert signaling receptivity_
(``receptivity''), _disliking penalty_ (``disliking''), and _minority populations_. 
The minority populations experiment essentially disliking 
experiment, but sets a minority of the population to have one trait (1 in code) 
and a majority of the population to have the opposite trait (-1 in code). In the
minority experiment, the fraction of the population in the minority is varied.

Below is more on each experiment and code examples showing how to run them.

The output dataframes can be used in the `plot_evolution`, `heatmap`, and,
for the minority experiments, `minority_diff_heatmap` imported from
`id_signaling.figures`. There are a number of 
keyword arguments that must be specified to do 
plotting for each experiment. Examples are given below.


## Covert signaling receptivity

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
