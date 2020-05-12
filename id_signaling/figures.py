import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_evolution(df, experiment='receptivity',
                   strategy='signaling',
                   exp_param_vals=[0.1, 0.25, 0.4],
                   homophily_vals=[0.1, 0.25, 0.4],
                   minority_subset=None,
                   savefig_path=None,
                   ax=None,
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

    ax = means.plot(lw=4, alpha=0.75, ax=ax, **plot_kwargs)

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

    return ax


def plot_coevolution(df, experiment, exp_param_vals, homophily_vals,
                     colors=['black', 'mediumorchid', 'dodgerblue'],
                     savefig_path=None, ax=None,
                     figsize=(7.5, 4), **plot_kwargs):

    # Plot N lines for given exp_param_vals and homophily_vals
    # representing evolution of signaling strategy in terms of
    # density of covert signalers.
    ax = plot_evolution(df, experiment=experiment,
                        exp_param_vals=exp_param_vals,
                        homophily_vals=homophily_vals,
                        figsize=figsize, color=colors, ax=ax)

    # Plot N lines for given exp_param_vals and homophily_vals
    # representing evolution of receiving strategy in terms of
    # density of covert signalers.
    plot_evolution(df, experiment=experiment,
                   exp_param_vals=exp_param_vals,
                   strategy='receiving',
                   homophily_vals=homophily_vals, ls='--',
                   figsize=figsize, color=colors, ax=ax)

    plt.title('Coevolution of sending\nand receiving strategies')

    plt.ylabel(r'$\rho_{cov}$ (solid)' '\n' r'$\rho_{ch}$ (dashed)',
               rotation=0, ha='right', size=14)

    plt.yticks(np.arange(0, 1.01, 0.25));

    # Hacky, written currently for up to 3 exp_param_vals and homophily vals.
    handles, labels = ax.get_legend_handles_labels()

    if experiment == 'disliking':
        exp_inset = r'$(d=\delta,~w)$'
    elif experiment == 'receptivity':
        exp_inset = r'$(r,~w)$'

    ax.legend(handles[:3], labels[:3], bbox_to_anchor=(1.01, 0.9), loc='upper left',
          ncol=1, title=exp_inset, borderaxespad=0, frameon=False, prop={'size': 12},
          title_fontsize=14)

    if savefig_path is not None:
        plt.savefig(savefig_path)


def heatmap(df, experiment='disliking', strategy='signaling',
            figsize=(6, 4.75), minority_subset=None, savefig_path=None,
            title=None, cmap='YlGnBu_r', **heatmap_kwargs):

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
    # final_means = means[means.timestep == means.timestep.max()]
    # print(final_means.stack().max())
    # print(final_means.stack())
    # print(final_means.head())


    plt.figure(figsize=figsize)

    ax = sns.heatmap(final_means.stack(), cmap=cmap, square=True,
                     vmin=0.0, vmax=1.0,
                     # cbar_kws={'label': f'Density of {strategy_inset.lower()}'},
                     **heatmap_kwargs
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
        relative_receptivity = np.sort(df.receptivity.unique() / 0.5)
        ax.set_xticklabels([f'{x:.1f}' for x in relative_receptivity], rotation=30)
    else:
        ax.set_xticklabels([f'{x:.2f}'
                            for x in np.sort(df.disliking.unique())], rotation=30)

    # ax.set_yticklabels([f'{y:.2f}' for y in np.arange(0.1, 0.46, 0.05)] + ['0.49', '0.50']);

    ax.set_yticklabels([f'{y:.2f}' for y in np.sort(df['homophily'].unique())])

    if title is not None:
        ax.set_title(title, size=14)
    if savefig_path is not None:
        plt.savefig(savefig_path)

def minority_diff_heatmap(df, strategy='signaling', savefig_path=None,
                          figsize=(6.35, 4.75), vmin=None, vmax=None,
                          title=None):

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

    pre = df[df.timestep == df.timestep.max()]

    gb_minority_mean = pre.groupby(['disliking', 'homophily', 'timestep'])[data_col_min].mean()
    minority_means = gb_minority_mean.unstack(level=(0, 1))

    gb_majority_mean = pre.groupby(['disliking', 'homophily', 'timestep'])[data_col_maj].mean()
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
                f'Difference in {strategy_inset.lower()}\n'
                f'prevalence (minority - majority)\n'
                # f'Minority - Majority',
            # 'ticks':
            #     np.arange(-0.2, 0.31, 0.1),
        }
    )

    ax.set_title(title)

    # Set size of colorbar title.
    ax.figure.axes[-1].yaxis.label.set_size(14)

    # Set size of colorbar tick labels.
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)

    ax.set_xlabel('Cost of disliking, $d=\delta$', size=15)

    # Clean up some other things.
    ax.invert_yaxis()
    # ax.set_yticklabels(['0.1', '0.25', '0.4'])
    ax.set_ylabel('Homophily, $w$', size=15)
    # ax.set_yticklabels([f'{y:.2f}' for y in np.arange(0, 0.46, 0.05)]);
    # ax.set_yticklabels([f'{y:.2f}' for y in np.arange(0.1, 0.46, 0.05)] + ['0.49', '0.50']);

    xvals = np.sort(pre.disliking.unique())
    yvals = np.sort(pre.homophily.unique())

    ax.set_xticklabels([f'{x:.2f}' for x in xvals], rotation=25)
    ax.set_yticklabels([f'{y:.2f}' for y in yvals])

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


def load_minority_dfs(directory='data/minority',
                      minority_vals=[0.05, 0.10, 0.15, 0.2,
                                     0.25, 0.3, 0.35, 0.4, 0.45]):

    minority_vals_labels = [f'{val:.2f}' for val in minority_vals]

    return [
        (label, pd.read_csv(os.path.join(directory, label, 'full.csv')))
        for label in minority_vals_labels
    ]


def covert_vs_minority_frac(minority_dfs, dislikings, homophily,
                            exclude_p05=True,  # Exclude data point by default.
                            ax=None, savefig_path=None):

    if exclude_p05:
        minority_dfs = minority_dfs[1:]  # For excluding 0.05 case.

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    minority_frac_strs = [el[0] for el in minority_dfs]

    line_styles = ['-', '--', ':']
    marker_styles = ['^', 'o', 's']

    for d_idx, disliking in enumerate(dislikings):

        # means_over_minority_frac = []
        # means_over_majority_frac = []
        minmaj_covert_mean_diff = []
        minmaj_covert_std_diff = []

        # std_over_minority_frac = []

        for minority_frac_str, df in minority_dfs:
            # First extract all N_trials final covert signaling proportions.
            pre = df[
                (df.timestep == 500) &
                (df.disliking == disliking) &
                (df.homophily == homophily)
            ]

            pre['min_maj_cov_diff'] = \
                pre.prop_covert_minority - pre.prop_covert_majority
            # print(pre['min_maj_cov_ratio'])
            # print(pre.min_maj_cov_ratio.mean())
            minmaj_covert_mean_diff.append(pre.min_maj_cov_diff.mean())

            std_final_cov_prop = pre.min_maj_cov_diff.std()
            minmaj_covert_std_diff.append(std_final_cov_prop)

            # minmaj_covert_std_diff = pre.prop_covert_minority.std()

            # means_over_minority_frac.append(mean_min_cov_prop)
            # means_over_majority_frac.append(mean_maj_cov_prop)
            # std_over_minority_frac.append(std_final_cov_prop)

        # majmin_covert_ratio = [min_ / maj_ for min_, maj_ in
        #                        zip(means_over_minority_frac, means_over_majority_frac)]
        # majmin_covert_ratio = [min_ / maj_ for min_, maj_ in
        #                        zip(means_over_minority_frac, means_over_majority_frac)]
        # majmin_covert_diff = [maj_ - min_ for min_, maj_ in
        #                        zip(means_over_minority_frac, means_over_majority_frac)]
        # ax.plot(means_over_minority_frac, color='black', ls=styles[d_idx],
        #         label=f'$d=\\delta={disliking:.2f}$')

        # ax.plot(majmin_covert_ratio, color='black', ls=styles[d_idx],
        #         label=f'$d=\\delta={disliking:.2f}$')
        # ax.plot(minmaj_covert_mean_ratio, color='black', ls=line_styles[d_idx],
        #         label=f'$d=\\delta={disliking:.2f}$')

        ax.plot(minmaj_covert_mean_diff, color='black', ls=line_styles[d_idx],
                marker=marker_styles[d_idx], mfc='white', mec='black', mew=1,
                label=f'$d=\\delta={disliking:.2f}$')
        # print(std_over_minority_frac)
        # ax.errorbar(range(len(minmaj_covert_mean_diff)),
        #             # means_over_minority_frac, yerr=std_over_minority_frac,
        #             minmaj_covert_mean_diff, yerr=minmaj_covert_std_diff,
        #             color='black',
        #             ls=line_styles[d_idx],
        #             label=f'$d=\\delta={disliking:.2f}$')

        ax.set_xticks(range(len(minority_frac_strs)))

        ax.set_xticklabels(minority_frac_strs, size=14, rotation=35)

        ylow = -0.3
        yhigh = 0.4
        ax.set_ylim(ylow, yhigh)

        yticks = np.arange(ylow, yhigh+0.01, 0.1)
        ax.set_yticks(yticks)
        # ax.set_yticklabels(labels=[f'{y:.2f}' for y in yticks], size=14)
        ax.tick_params('y', labelsize=14)
        ax.set_ylabel(r'$\rho^{minor}_{cov,T} - \rho^{major}_{cov,T}$',
                      size=15)
        ax.set_xlabel(r'$\rho^{minor}$', size=15)

        ax.grid(axis='y')

        ax.legend()

        ax.set_title(f'$w={homophily:.2f}$', size=14)

        if savefig_path is not None:
            plt.savefig(savefig_path)


def similarity_threshold(dfs, thresholds=np.arange(0.1, 1.1, 0.1),
                         dislikings=[0.05, 0.25, 0.45], homophily=0.2,
                         ax=None, savefig_path=None, ylow=0.2, yhigh=1.0,
                         legend=False, xlabel=True, ylabel=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    line_styles = ['-', '--', ':']
    marker_styles = ['^', 'o', 's']

    for d_idx, disliking in enumerate(dislikings):

        mean_covert_prevalence = []

        for df_idx, df in enumerate(dfs):

            mean_covert_prevalence.append(
                df[
                    (df.timestep == 500) &
                    (df.disliking == disliking) &
                    (np.isclose(df.homophily, homophily))
                ].prop_covert.mean()
            )

        ax.plot(mean_covert_prevalence, color='black', ls=line_styles[d_idx],
                marker=marker_styles[d_idx], mfc='white', mec='black', mew=1,
                label=f'$d=\\delta={disliking:.2f}$')

    if legend:
        ax.legend()

    ax.set_title(f'$w={homophily:.1f}$')

    ax.set_ylim(ylow, yhigh)
    ax.grid(axis='y')

    if xlabel:
        ax.set_xlabel('Similarity threshold, $S$', size=14)
    if ylabel:
        ax.set_ylabel('Covert prev., $\\rho_{cov,t=T}$', size=14)

    ax.set_xticks(range(1, 10, 2))
    ax.set_xticklabels([f'{threshold:1.1f}' for threshold in thresholds[1::2]])
    # ax.set_yticks(np.arange(0.2, 1.1, 0.1))


def invasion_heatmaps(disliking_df, recept_df,
                      cmap=sns.cubehelix_palette(
                          50, hue=0.05, rot=0, light=0.0,
                          dark=0.9, as_cmap=True
                      ),
                      invading='covert',
                      vmin=-0.05, vmax=0.05,
                      base_filename='reports/Figures/invasion',
                      figsize=(8, 5),
                      save_path=None):
    '''
    Plot 3x2 heatmaps in subplots. 3-element rows are each for one of the
    two experiments, with disliking penalty or covert receptivity on the
    y-axis and homophily on the x-axis.

    I suspect there is a smart way to do this, but I'm repeating a lot of
    code from overt/covert to generous/churlish plotting. It's just the
    fastest way right now that shouldn't be too onerous.
    '''
    timesteps = 500

    fig, axes = plt.subplots(2, 3, sharex=True, figsize=figsize)

    # Step 1: do this for covert invading and generalize from there.
    rate_dfs = []
    if invading in ('overt', 'covert'):

        if invading == 'covert':
            init_cov = 0.05
        else:
            init_cov = 0.95

        init_churs = [0.05, 0.5, 0.95]

        # final_dis = disliking_df[disliking_df.timestep == timesteps]
        # final_rec = recept_df[recept_df.timestep == timesteps]
        for exp_idx, name_df in enumerate(
                            [("Disliking penalty", disliking_df),
                             ("Relative\ncovert receptivity", recept_df)]
                        ):

            for chur_idx, init_chur in enumerate(init_churs):

                name = name_df[0]
                df = name_df[1]

                # ax.plot((exp_idx + chur_idx) * np.arange(10))
                df_lim = df[(df.initial_prop_covert == init_cov) &
                            (df.initial_prop_churlish == init_chur)]

                rate_df = _one_invasion_heatmap(
                    axes, name, df_lim, invading, init_cov, init_chur, exp_idx,
                    chur_idx, timesteps, cmap)
                rate_dfs.append(rate_df)

    elif invading in ('churlish', 'generous'):

        if invading == 'churlish':
            init_chur = 0.05
        else:
            init_chur = 0.95

        init_covs = [0.05, 0.5, 0.95]

        for exp_idx, name_df in enumerate(
                        [("Disliking penalty", disliking_df),
                         ("Relative\ncovert receptivity", recept_df)]
                    ):

            for cov_idx, init_cov in enumerate(init_covs):

                name = name_df[0]
                df = name_df[1]

                df_lim = df[(df.initial_prop_covert == init_cov) &
                            (df.initial_prop_churlish == init_chur)]

                rate_df = _one_invasion_heatmap(
                    axes, name, df_lim, invading, init_cov, init_chur, exp_idx,
                    cov_idx, timesteps, cmap)
                rate_dfs.append(rate_df)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return rate_dfs


def _one_invasion_heatmap(axes, name, df_lim, invading, init_cov,
                          init_chur, exp_idx, chur_cov_idx, timesteps,
                          cmap, low_success_prevalence=0.05):
    '''

    Arguments:
        low_success_prevalence (float): non-inclusive lower limit on how
            prevalent a strategy must be for invasion to be considered
            successful. 0, for example, would be mean any non-zero population
            can be consdered to have successfully invaded. The default, 0.05,
            means invasion is successful if the prevalence is greater than
            1/20.
    '''

    final = df_lim[df_lim.timestep == timesteps]

    # Create a new boolean column marking if invasion was
    # successful.
    if invading == 'covert':
        final['success'] = final.prop_covert > low_success_prevalence
    elif invading == 'overt':
        final['success'] = final.prop_covert < 1.0 - low_success_prevalence
    elif invading == 'churlish':
        final['success'] = final.prop_churlish > low_success_prevalence
    elif invading == 'generous':
        final['success'] = final.prop_churlish < 1.0 - low_success_prevalence
    else:
        raise RuntimeError(f'{invading} strategy not recognized')

    # Set correct column name to aggregate over.
    colname = ('disliking', 'receptivity')[exp_idx]

    rate = final.groupby(
        [colname, 'homophily']
    ).agg({'success': np.mean})

    # import ipdb
    # ipdb.set_trace()

    ax = axes[exp_idx, chur_cov_idx]

    shrink = 0.6
    if chur_cov_idx != 2:
        # cbar=False
        cbar = True
        cbar_kws = {'shrink': shrink}
    else:
        cbar = True
        cbar_kws = {'label': 'Invasion success ratio', 'shrink': shrink}

    sns.heatmap(
        rate['success'].unstack(),
        cmap=cmap, square=True, vmin=0, vmax=1,
        ax=ax, cbar=cbar, cbar_kws=cbar_kws
    )

    ax.set_xticklabels(
        [f'{x:.1f}' for x in np.sort(df_lim.homophily.unique())],
        rotation=0
    )

    if colname == 'disliking':
        if chur_cov_idx == 0:
            labs = [f'{y:.1f}' for y in np.sort(df_lim[colname].unique())]
            ax.set_yticklabels(
                labs,
                rotation=0
            )
        else:
            ax.set_yticklabels(['']*6)
    else:
        if chur_cov_idx == 0:
            ax.set_yticklabels(
                [f'{y:.1f}' for y in np.sort(df_lim[colname].unique()) / 0.5],
                rotation=0
            )
        else:
            ax.set_yticklabels(['']*6)

    if exp_idx == 0:
        if invading in ['overt', 'covert']:
            ax.set_title(f'$\\rho_{{ch,0}} = {init_chur}$', size=12)
        else:
            ax.set_title(f'$\\rho_{{cov,0}} = {init_cov}$', size=12)

    if chur_cov_idx == 0:
        ax.set_ylabel(name, size=12)
    else:
        ax.set_ylabel('')

    if exp_idx == 1:
        ax.set_xlabel('Homophily', size=12)
    else:
        ax.set_xlabel('')

    ax.invert_yaxis()

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return rate
