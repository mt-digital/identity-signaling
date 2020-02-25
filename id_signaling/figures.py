

import matplotlib.pyplot as plt


def plot_evolution(df, experiment='receptivity',
                   strategy='signaling', parameter_vals=[0.1, 0.25, 0.4],
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
        parameter_vals (list): Values of the homophily and disliking/receiving
            parameters to plot, used for subsetting input data frame.
        plot_kwargs (dict): Additional arguments for passing to plotting
            functions, e.g., `lw=3` to set line width.
    '''

    # Set up plot annotations depending on which kind of experiment
    # and strategy is plotted.
    if experiment == 'disliking':
        exp_inset = '$(d=\delta,~w)$'
        exp_col = 'dislike_penalty'
    elif experiment == 'receptivity':
        exp_inset = '$(r/R,~w)$'
        exp_col = 'receptivity'
    else:
        raise RuntimeError(f'{experiment} not recognized')

    if strategy == 'signaling':
        strategy_inset = 'Covert signalers'
        data_col = 'prop_covert'
    elif strategy == 'receiving':
        strategy_inset = 'Churlish receivers'
        data_col = 'prop_churlish'
    else:
        raise RuntimeError(f'{strategy} not recognized')

    # Build title from `inset`s above.
    title = f'{strategy_inset} over time for various {exp_inset} pairs'

    # Limit data to specified parameter values.
    df = df[
        df['homophily'].isin(parameter_vals) &
        df[exp_col].isin(parameter_vals)
    ]

    # Some lines I hacked together to take mean over trials and get something
    # easy to plot.
    gb_mean = df.groupby([exp_col, 'homophily', 'timestep'])[data_col].mean()
    means = gb_mean.unstack(level=(0, 1))
    # plt.figure(figsize=(7.5, 4.5))
    # XXX Hack to test if this works for weird problem with churlish plot.
    # if 'figsize' in plot_kwargs:
    #     del plot_kwargs['figsize']

    means.plot(lw=4, alpha=0.75, **plot_kwargs)

    # Put legend outside of plot with label determined by experiment type.
    plt.legend(bbox_to_anchor=(1.01, 0.9), loc='center left', ncol=1, title=exp_inset,
               borderaxespad=0, frameon=False, prop={'size': 12})

    # plt.legend(ncol=3, title=exp_inset,
    #            borderaxespad=0, frameon=False, prop={'size': 12})

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
