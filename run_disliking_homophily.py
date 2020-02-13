from id_signaling.experiment import trials_dislikepen_homophily

import os

data_dir = 'output_data'

disliking_penalties = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
homophilies = receptivities

df_full = None

for d in disliking_penalties:
    for w in homophilies:

        df = trials_dislikepen_homophily(r, w, n_iter=200, n_trials=100)

        if df_full is None:
            df_full = df
        else:
            df_full = df_full.append(df)

df_full.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
