from id_signaling.experiment import trials_receptivity_homophily

import os

data_dir = 'output_data'

receptivities = [0.2, 0.4]
homophilies = [0.0, 0.25]

df_full = None

for r in receptivities:
    for w in homophilies:

        df = trials_receptivity_homophily(r, w, n_iter=100, n_trials=10)

        if df_full is None:
            df_full = df
        else:
            df_full = df_full.append(df)

df_full.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
