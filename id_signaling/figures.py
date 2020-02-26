

import matplotlib.pyplot as plt


def plot_evolution(df, experiment='receptivity',
                   strategy='signaling',
                   exp_param_vals=[0.1, 0.25, 0.4],
                   homophily_vals=[0.1, 0.25, 0.4],
                   minority_subset=None,
                   savefig_path=None,
                   **plot_kwargs):
    '''
    Arguments:
        df (pandas.DataFrame):
        experiment (str): Either 'receptivity' or 'disliking' for now,
            indicating whether receptivity and homophily, or disliking and
            homophily, were co-varied in the experiment.
        strategy (str): Either 'signaling' or 'receiving', specifying which
            output variable to use. If 'signaling', the proportion of
            covert signalers is plotted. If 'receiving', the proportion of
            churlish receivers is plotted.
        minority_subset (str or None): If given, should be 'minority' or
            'majority' to plot either minority or majority evolution of
            signaling or receiving strategy.
        parameter_vals (list): Values of the homophily and disliking/receiving
            parameters to plot, used for subsetting input data frame.
        plot_kwargs (dict): Additional arguments for passing to plotting
            functions, e.g., `lw=3` to set line width.
    '''

    # Set up plot annotations depending on which kind of experiment
    # and strategy is plotted.
    if experiment == 'disliking':
        exp_inset = '$(d=\delta,~w)$'
        exp_col = 'disliking'
    elif experiment == 'receptivity':
        exp_inset = '$(r/R,~w)$'
        exp_col = 'receptivity'
    else:
        raise RuntimeError(f'{experiment} not recognized')

    # Plot non-minority/majority subsetted data.
    if strategy == 'signaling':
        strategy_inset = 'Covert signalers'
        data_col = 'prop_covert'
    elif strategy == 'receiving':
        strategy_inset = 'Churlish receivers'
        data_col = 'prop_churlish'
    else:
        raise RuntimeError(f'{strategy} not recognized')

    # Modify title/ylabel inset and data column of interest if subsetting.
    if minority_subset == 'minority':
        strategy_inset += ' minority'
        data_col += '_minority'
    elif minority_subset == 'majority':
        strategy_inset += ' majority'
        data_col += '_majority'

    # Build title from `inset`s above.
    title = f'{strategy_inset} over time for various {exp_inset} pairs'

    # Limit data to specified parameter values.
    df = df[
        df['homophily'].isin(homophily_vals) &
        df[exp_col].isin(exp_param_vals)
    ]

    # Some lines I hacked together to take mean over trials and get something
    # easy to plot.
    gb_mean = df.groupby([exp_col, 'homophily', 'timestep'])[data_col].mean()
    means = gb_mean.unstack(level=(0, 1))

    means.plot(lw=4, alpha=0.75, **plot_kwargs)

    # Put legend outside of plot with label determined by experiment type.
    plt.legend(bbox_to_anchor=(1.01, 0.9), loc='upper left', ncol=1, title=exp_inset,
               borderaxespad=0, frameon=False, prop={'size': 12})

    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.ylabel(f'Proportion of {strategy_inset.lower()}', size=16)
    plt.xlabel('Timestep', size=16)
    plt.title(title, size=14)

    if savefig_path is not None:
        plt.savefig(savefig_path)


def heatmap(df, strategy='signal'):
    pass


def covert_churlish_regression(df):
    # Get pairs of final (prop_covert, prop_churlish) for each trial and
    # perform/plot regression of prop_covert vs. prop_churlish.
    pass
