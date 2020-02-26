import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        strategy_inset += ' in minority'
        data_col += '_minority'
    elif minority_subset == 'majority':
        strategy_inset += ' in majority'
        data_col += '_majority'

    # Build title from `inset`s above.
    title = f'{strategy_inset}\nover time for various {exp_inset} pairs'

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
               borderaxespad=0, frameon=False, prop={'size': 12}, title_fontsize=14)

    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.ylabel(f'Proportion of\n{strategy_inset.lower()}', size=16)
    plt.xlabel('Timestep', size=16)
    plt.title(title, size=14)

    if savefig_path is not None:
        plt.savefig(savefig_path)


def heatmap(df, experiment='disliking', strategy='signaling',
            figsize=(6, 4.75), minority_subset=None, savefig_path=None):

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

    # Some lines I hacked together to take mean over trials and get something
    # easy to plot.
    gb_mean = df.groupby([exp_col, 'homophily', 'timestep'])[data_col].mean()
    means = gb_mean.unstack(level=(0, 1))

    final_means = means[means.index == means.index[-1]]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(final_means.stack(), cmap='YlGnBu_r', square=True,
                    cbar_kws={'label': f'Density of {strategy_inset.lower()}'},
                    )

    # Set size of colorbar title.
    ax.figure.axes[-1].yaxis.label.set_size(14)

    # Set size of colorbar tick labels.
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)

    # Clean up some other things.
    ax.invert_yaxis()
    # ax.set_yticklabels(['0.1', '0.25', '0.4'])
    ax.set_ylabel('Homophily, $w$', size=15)
    if experiment == 'receptivity':
        ax.set_xlabel('Relative covert receptivity, $r/R$', size=15)
    elif experiment == 'disliking':
        ax.set_xlabel('Cost of disliking, $d=\delta$', size=15)


    if experiment == 'receptivity':
        relative_receptivity = df.receptivity.unique() / 0.5
        relative_receptivity.sort()
        ax.set_xticklabels(relative_receptivity)

    ax.set_yticklabels([f'{y:.2f}' for y in np.arange(0, 0.46, 0.05)]);

    if savefig_path is not None:
        plt.savefig(savefig_path)

def minority_diff_heatmap(df, strategy='signaling', savefig_path=None,
                          figsize=(6.35, 4.75), vmin=None, vmax=None):

    if strategy == 'signaling':
        strategy_inset = 'Covert signalers'
        data_col = 'prop_covert'
    elif strategy == 'receiving':
        strategy_inset = 'Churlish receivers'
        data_col = 'prop_churlish'
    else:
        raise RuntimeError(f'{strategy} not recognized')

    data_col_min = data_col + '_minority'
    data_col_maj = data_col + '_majority'

    gb_minority_mean = df.groupby(['disliking', 'homophily', 'timestep'])[data_col_min].mean()
    minority_means = gb_minority_mean.unstack(level=(0, 1))

    gb_majority_mean = df.groupby(['disliking', 'homophily', 'timestep'])[data_col_maj].mean()
    majority_means = gb_majority_mean.unstack(level=(0, 1))

    final_min_means = minority_means[minority_means.index == minority_means.index[-1]]
    final_maj_means = majority_means[minority_means.index == majority_means.index[-1]]

    diff = final_min_means - final_maj_means

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        diff.stack(), square=True, vmin=vmin, vmax=vmax,
        cmap=sns.diverging_palette(10, 220, sep=80, n=10), #, center='dark'),
        cbar_kws={
            'label':
                f'Difference in density\n'
                f'of {strategy_inset.lower()}\n'
                f'Minority - Majority',
            'ticks':
                np.arange(-0.2, 0.31, 0.1),
        }
    )

    # Set size of colorbar title.
    ax.figure.axes[-1].yaxis.label.set_size(14)

    # Set size of colorbar tick labels.
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)

    ax.set_xlabel('Cost of disliking, $d=\delta$', size=15)

    # Clean up some other things.
    ax.invert_yaxis()
    # ax.set_yticklabels(['0.1', '0.25', '0.4'])
    ax.set_ylabel('Homophily, $w$', size=15)
    ax.set_yticklabels([f'{y:.2f}' for y in np.arange(0, 0.46, 0.05)]);

    if savefig_path is not None:
        plt.savefig(savefig_path)

def covert_churlish_regression(df, # exp_param_vals, homophily_vals,
                               experiment='receptivity',
                               minority_subset=None, savefig_path=None,
                               **kwargs):
    # Get pairs of final (prop_covert, prop_churlish) for each trial and
    # perform/plot regression of prop_covert vs. prop_churlish.
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

    # Limit data to specified parameter values.
    df = df[
        (df['timestep'] == df['timestep'].max())
    ]

    # Extract data from dataframe.
    post = '_' + minority_subset if minority_subset is not None else ''
    xcol = 'prop_churlish' + post
    ycol = 'prop_covert' + post

    plt.figure(**kwargs)

    x = df[xcol]
    y = df[ycol]

    from scipy import stats

    sns.regplot(x, y, line_kws=dict(color='r'),
                scatter_kws=dict(lw=0, s=45, alpha=0.125))
    plt.ylabel('Covert signaler proportion', size=16)
    plt.xlabel('Churlish receiver proportion', size=16)

    pearson_coef, p_value = stats.pearsonr(x, y) #define the columns to perform calculations on

    plt.title(f'{experiment.title()}, all parameter combinations\nPearson correlation coefficient: {pearson_coef:.2f}\nP-value: {p_value:.2e}')

    if savefig_path is not None:
        plt.savefig(savefig_path)
