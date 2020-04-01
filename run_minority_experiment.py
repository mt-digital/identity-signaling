from id_signaling.experiment import run_experiments

import sys
import os

# XXX Change this to change the minority fraction.
minority_trait_frac = 0.05

data_dir = 'output_data'

disliking_penalties = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
homophilies = disliking_penalties[2:] + [0.49, 0.5]

df = run_experiments(disliking_penalties, homophilies,
    minority_trait_frac=minority_trait_frac,
    experiment='disliking',
    n_iter=200,
    n_trials=100
)

df.to_csv(os.path.join(data_dir, f'minorities_{minority_trait_frac}.csv'),
          index=False)
