import os

import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection

import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import utils.utils_vis as utils_vis
import scripts.config as config
from visualizations.vis_config import *



## Read parameters from config
data_dir           = config.data_dir
results_dir        = config.results_dir
vis_dir            = config.vis_dir
session_data_dict  = config.session_data_dict
session_data_names = config.session_data_names

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
data_formats       = config.data_formats
label_formats      = config.label_formats
trial_filters      = config.trial_filters
train_test_options = config.train_test_options

random_states      = config.random_states
n_folds            = config.n_folds
ns_states          = config.ns_states
ns_discrete_states = config.ns_discrete_states
ns_iters           = config.ns_iters
window_sizes       = config.window_sizes

model_types        = config.model_types
dynamics_classes   = config.dynamics_classes
emission_classes   = config.emission_classes
init_types         = config.init_types
subspace_types     = config.subspace_types
alphas             = config.alphas

n_ns_continuous_states = len(ns_states)
n_ns_discrete_states   = len(ns_discrete_states)
n_ns_iters             = len(ns_iters)



alpha_point = 0.8
size_point  = 100
alpha_line_thin  = 0.3
alpha_line       = 0.6
alpha_line_thick = 0.9
size_line_thin  = 1
size_line       = 3
size_line_thick = 5
label_fontsize = 18



theme_coral_light = (255/255, 204/255, 204/255)
theme_coral_mid   = (247/255, 156/255, 156/255)
theme_coral_dark  = (232/255, 107/255, 107/255)
theme_green_light = (206/255, 242/255, 238/255) 
theme_green_mid   = (120/255, 204/255, 194/255)
theme_green_dark  = (74/255,  189/255, 175/255)
theme_blue_light = sns.color_palette('Blues')[0]
theme_blue_mid   = sns.color_palette('Blues')[2]
theme_blue_dark  = sns.color_palette('Blues')[3]
theme_orange_light = sns.color_palette('Oranges')[0]
theme_orange_mid   = sns.color_palette('Oranges')[2]
theme_orange_dark  = sns.color_palette('Oranges')[3]

corals  = [theme_coral_light, theme_coral_mid, theme_coral_dark]
greens  = [theme_green_light, theme_green_mid, theme_green_dark]
blues   = [theme_blue_light, theme_blue_mid, theme_blue_dark]
oranges = [theme_orange_light, theme_orange_mid, theme_orange_dark]


trial_filter_colors_palettes = {
    'fast': blues,
    'slow': oranges,
    'far':  corals,
    'near': greens,
}

trial_filter_colors = {
    'fast': blues[1],
    'slow': oranges[1],
    'far':  corals[1],
    'near': greens[1],
}

discrete_state_cmap = sns.color_palette('spring', as_cmap=True)

trial_filter_cmaps = {
    # 'slow': sns.color_palette('Oranges', as_cmap=True),
    # 'fast': sns.color_palette('Blues', as_cmap=True),

    # 'slow': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_slow', [sns.color_palette('Oranges')[0], sns.color_palette('Oranges')[3]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_fast', [sns.color_palette('Blues')[3],   sns.color_palette('Blues')[0]]),

    # 'slow': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_slow', [sns.color_palette('Oranges')[3], sns.color_palette('Oranges')[0]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_fast', [sns.color_palette('Blues')[3],   sns.color_palette('Blues')[0]]),
    'slow': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'fast': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'near': mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_near', [greens[2], greens[0]]),
    'far':  mcolors.LinearSegmentedColormap.from_list('trial_filter_cmap_far',  [corals[2], corals[0]]),
    # 'fast': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
    'gray_r_cmap': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    'gray_cmap': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
}

trial_filter_colors_by_discrete_state = {
    # 'slow': [sns.color_palette('Oranges')[0], sns.color_palette('Oranges')[3]],
    # 'fast': [sns.color_palette('Blues')[0], sns.color_palette('Blues')[3]],

    'slow': [
        sns.color_palette('Oranges')[3], 
        sns.color_palette('Oranges')[0],
        sns.color_palette('Oranges')[1],
        sns.color_palette('Oranges')[2],
        sns.color_palette('Oranges')[4],
        sns.color_palette('Oranges')[5],
        sns.color_palette('Reds')[3],
        sns.color_palette('Reds')[0],
        sns.color_palette('Reds')[1],
        sns.color_palette('Reds')[2],
        sns.color_palette('Reds')[4],
        sns.color_palette('Reds')[5],
        sns.color_palette('Greys')[0],
        sns.color_palette('Greys')[1],
        sns.color_palette('Greys')[2],
        sns.color_palette('Greys')[3]],
    'fast': [
        sns.color_palette('Blues')[3], 
        sns.color_palette('Blues')[0],
        sns.color_palette('Blues')[1],
        sns.color_palette('Blues')[2],
        sns.color_palette('Blues')[4],
        sns.color_palette('Blues')[5],
        sns.color_palette('Greens')[3],
        sns.color_palette('Greens')[0],
        sns.color_palette('Greens')[1],
        sns.color_palette('Greens')[2],
        sns.color_palette('Greens')[4],
        sns.color_palette('Greens')[5],
        sns.color_palette('Purples')[0],
        sns.color_palette('Purples')[1],
        sns.color_palette('Purples')[2],
        sns.color_palette('Purples')[3]],
    'near': [greens[2], greens[0], greens[1]],
    'far': [corals[2], corals[0], corals[1]],
}

metric_plot_colors = {
    'rSLDS slow': '#C93735',
    'rSLDS fast': '#E59693',
    'LDS slow':   '#333A8C',
    'LDS fast':   '#989CC8',
    'PCA slow':   '#F8AB61',
    'PCA fast':   '#FCD7AF',
}

# discrete_state_colors = [
#     '#FD8D61',
#     '#5D63AE',
# ]

# discrete_state_colors = [
#     '#5D63AE',
#     '#FD8D61',
# ]

discrete_state_colors = sns.color_palette('Set2')
discrete_state_colors_with_PCA = [mcolors.to_rgb('skyblue')] + discrete_state_colors
continuous_state_colors = sns.color_palette('rocket', n_colors=n_ns_continuous_states)

trial_filter_name_conversion = {
    'fast': 'Ballistic',
    'slow': 'Sustained',
    'near': 'Near',
    'far':  'Far',
}

time_gradient_cmap = sns.color_palette('winter_r', as_cmap=True)
# time_gradient_cmap = sns.color_palette('hsv', as_cmap=True)
# time_gradient_cmap = sns.color_palette('summer', as_cmap=True)

caltech_orange = '#FF6C0C'

target_color_palette_8  = sns.color_palette('hls', 8)

discrete_state_quiver_cmaps = [
    sns.color_palette('crest_r',   as_cmap=True),
    sns.color_palette('flare_r',   as_cmap=True),
    sns.color_palette('rocket',    as_cmap=True), 
    sns.color_palette('mako',      as_cmap=True),
    sns.color_palette('viridis',   as_cmap=True),
    sns.color_palette('plasma',    as_cmap=True),
    sns.color_palette('inferno',   as_cmap=True),
    sns.color_palette('magma',     as_cmap=True),
    sns.color_palette('cividis',   as_cmap=True),
    sns.color_palette('cubehelix', as_cmap=True),
    sns.color_palette('Spectral',  as_cmap=True),
    sns.color_palette('coolwarm',  as_cmap=True),
    sns.color_palette('bwr',       as_cmap=True),
    sns.color_palette('GnBu',      as_cmap=True),
    sns.color_palette('BuPu',      as_cmap=True),
    sns.color_palette('YlGnBu',    as_cmap=True),
]

elbos_cmaps = {
    'rSLDS': sns.color_palette('rocket', len(ns_states)),
    'LDS':   sns.color_palette('mako',   len(ns_states)),
}



def plot_decoding_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap',
    show_tick_labels=True):

    print('Plotting decoding results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_decoding_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    PCA_decoding_name = '_'.join(map(str, [x for x in [
        'decoding',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        'PCA'] if x is not None]))
    

    rSLDS_decoding_slow_all = []
    rSLDS_decoding_fast_all = []
    PCA_decoding_slow_all   = []
    PCA_decoding_fast_all   = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_decoding = np.load(os.path.join(session_results_dir, rSLDS_decoding_name + '.npz'))
        rSLDS_decoding_slow = rSLDS_decoding['decoding_errors_slow']
        rSLDS_decoding_fast = rSLDS_decoding['decoding_errors_fast']
        # PCA_decoding = np.load(os.path.join(session_results_dir, PCA_decoding_name + '.npz'))
        # PCA_decoding_slow = PCA_decoding['decoding_errors_slow']
        # PCA_decoding_fast = PCA_decoding['decoding_errors_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_decoding_slow = rSLDS_decoding_slow[..., -1]
        rSLDS_decoding_fast = rSLDS_decoding_fast[..., -1]
        # PCA_decoding_slow = PCA_decoding_slow
        # PCA_decoding_fast = PCA_decoding_fast

        rSLDS_decoding_slow_all.append(rSLDS_decoding_slow)
        rSLDS_decoding_fast_all.append(rSLDS_decoding_fast)
        # PCA_decoding_slow_all.append(PCA_decoding_slow)
        # PCA_decoding_fast_all.append(PCA_decoding_fast)

    ## Stack all sessions
    rSLDS_decoding_slow_all = np.stack(rSLDS_decoding_slow_all, axis=0)
    rSLDS_decoding_fast_all = np.stack(rSLDS_decoding_fast_all, axis=0)
    # PCA_decoding_slow_all = np.stack(PCA_decoding_slow_all, axis=0)
    # PCA_decoding_fast_all = np.stack(PCA_decoding_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_decoding_slow_all_mean = np.mean(rSLDS_decoding_slow_all, axis=(0, 2, 3))
    rSLDS_decoding_fast_all_mean = np.mean(rSLDS_decoding_fast_all, axis=(0, 2, 3))
    # PCA_decoding_slow_all_mean = np.mean(PCA_decoding_slow_all, axis=(0, 2, 3))
    # PCA_decoding_fast_all_mean = np.mean(PCA_decoding_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_decoding_slow_all_se = np.std(rSLDS_decoding_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_decoding_fast_all_se = np.std(rSLDS_decoding_fast_all, axis=(0, 2, 3)) / correction_factor
    # PCA_decoding_slow_all_se = np.std(PCA_decoding_slow_all, axis=(0, 2, 3)) / correction_factor
    # PCA_decoding_fast_all_se = np.std(PCA_decoding_fast_all, axis=(0, 2, 3)) / correction_factor

    ## Concatenate all decoding results
    # decoding_slow_mean = np.concatenate([PCA_decoding_slow_all_mean[:, :, None], rSLDS_decoding_slow_all_mean], axis=-1)
    # decoding_fast_mean = np.concatenate([PCA_decoding_fast_all_mean[:, :, None], rSLDS_decoding_fast_all_mean], axis=-1)
    # decoding_slow_se = np.concatenate([PCA_decoding_slow_all_se[:, :, None], rSLDS_decoding_slow_all_se], axis=-1)
    # decoding_fast_se = np.concatenate([PCA_decoding_fast_all_se[:, :, None], rSLDS_decoding_fast_all_se], axis=-1)

    decoding_slow_mean = rSLDS_decoding_slow_all_mean
    decoding_fast_mean = rSLDS_decoding_fast_all_mean
    decoding_slow_se = rSLDS_decoding_slow_all_se
    decoding_fast_se = rSLDS_decoding_fast_all_se

    decoding_results = {
        trial_filters[1]:
            {
                'rSLDS_mean': rSLDS_decoding_slow_all_mean,
                # 'PCA_mean': PCA_decoding_slow_all_mean,
                'all_mean': decoding_slow_mean,
                'rSLDS_se': rSLDS_decoding_slow_all_se,
                # 'PCA_se': PCA_decoding_slow_all_se,
                'all_se': decoding_slow_se,
            },
        trial_filters[0]:
            {
                'rSLDS_mean': rSLDS_decoding_fast_all_mean,
                # 'PCA_mean': PCA_decoding_fast_all_mean,
                'all_mean': decoding_fast_mean,
                'rSLDS_se': rSLDS_decoding_fast_all_se,
                # 'PCA_se': PCA_decoding_fast_all_se,
                'all_se': decoding_fast_se,
            }
    }
    
    vmin = 0
    vmax = 90

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1
    
    ## Plot results
    if form == 'scatter':

        ## Use PCA results as baseline
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # for i_tf, trial_filter in enumerate(trial_filters):

        #     decoding_results_rSLDS = decoding_results[trial_filter]['rSLDS'][train_or_test_id, ...]
        #     decoding_results_PCA   = decoding_results[trial_filter]['PCA'][train_or_test_id, ...]
        #     decoding_results_ = decoding_results_rSLDS - decoding_results_PCA[:, None]

        #     for i_ds in range(len(ns_discrete_states)):
                
        #         axs[i_tf].axhline(
        #             y=0,
        #             color='black',
        #             lw=size_line_thin,
        #             alpha=alpha_line_thin,
        #             linestyle='--')

        #         axs[i_tf].scatter(
        #             ns_states, 
        #             decoding_results_[:, i_ds], 
        #             color=discrete_state_colors[i_ds],
        #             alpha=alpha_line,
        #             s=size_point,
        #             )
                
        #         axs[i_tf].set_xticks(ns_states)

        ## Not using PCA results as baseline
        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):

            decoding_results_ = decoding_results[trial_filter]['all_mean'][train_or_test_id, ...]

            ## Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                decoding_results_, 
                decoding_results[trial_filter]['all_se'][train_or_test_id, ...],
                ns_states, 
                ns_discrete_states,
                includes_pca=True,
                higher_is_better=False)

            # axs[i_tf].axhline(
            #     y=performance_reference,
            #     color='black',
            #     lw=size_line_thin,
            #     alpha=alpha_line_thin,
            #     linestyle='--')

            for i_ds in reversed(range(len(ns_discrete_states))):

                ## Introduce jitter
                x = ns_states + np.random.uniform(-0.1, 0.1, size=len(ns_states))
                y = decoding_results_[:, i_ds]

                axs[i_tf].scatter(
                    x, 
                    y, 
                    color=discrete_state_colors[i_ds],
                    alpha=alpha_line,
                    s=2,
                    edgecolors='none',
                )

            xticks_range = [s for s in ns_states if 2 <= s <= 20]
            axs[i_tf].set_xticks(xticks_range)
            axs[i_tf].set_yticks([25, 50, 75])

            if show_tick_labels:
                tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(xticks_range)]
                axs[i_tf].set_xticklabels(tick_labels)
            else:
                axs[i_tf].set_xticklabels([])
                axs[i_tf].set_yticklabels([])

            axs[i_tf].set_ylim(vmin, vmax)
            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            # axs[i_tf].set_ylabel('Absolute angle error (degrees)', fontsize=5)

            ## Remove top and right spines
            axs[i_tf].spines['top'].set_visible(False)
            axs[i_tf].spines['right'].set_visible(False)


        # fig.subplots_adjust(bottom=0.2)

        # Define your custom markers for the legend
        # custom_markers = [
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[0], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[1], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[2], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[3], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[4], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[5], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[6], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[7], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line),
        #     Line2D([0], [0], color=discrete_state_colors_with_PCA[8], marker='o', linestyle='None', markeredgecolor='none', markersize=10, alpha=alpha_line)
        # ]

        # # Your legend labels remain the same
        # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

        # # Create the legend with the custom markers
        # plt.legend(
        #     custom_markers, 
        #     legend_labels, 
        #     title='# of Discrete States',
        #     loc='upper center', 
        #     bbox_to_anchor=(-0.2, -0.15), 
        #     fancybox=False, 
        #     shadow=False, 
        #     ncol=9
        # )


    elif form == 'swarm':
    
        fig, axs = plt.subplots(1, 2, figsize=(30, 6))
    
        for i_tf, trial_filter in enumerate(trial_filters):

            decoding_results_ = decoding_results[trial_filter]['all'][train_or_test_id, ...]
    
            # Prepare data for seaborn
            data_list = []
            for i_ds in range(len(ns_discrete_states) + 1):
                for ns_state, result in zip(ns_states, decoding_results_[:, i_ds]):
                    data_list.append({
                        'ns_state': ns_state,
                        'decoding_result': result,
                        'discrete_state': i_ds
                    })
                    
            data = pd.DataFrame(data_list)
            
            # Plot using seaborn
            sns.swarmplot(
                x='ns_state', 
                y='decoding_result', 
                hue='discrete_state', 
                data=data, 
                ax=axs[i_tf],
                palette=discrete_state_colors_with_PCA
            )

            axs[i_tf].set_ylim(10, 30)


    fig.tight_layout()

    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'decoding',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig)


def drop_and_truncate(
        arr,              # your decoding array, shape (..., n_trials, T_max)
        trial_lengths,    # 1-D array, len = n_trials
        drop_shortest_pct # e.g. 10  →  drop the shortest 10 %
):
    """
    1.  Drop trials whose length is below the requested percentile.
    2.  Optionally (see `crop`) crop the time axis so every kept trial
        is fully defined (no all-NaN tail).
    Returns
    -------
    arr_kept    : padded array with shorter trial axis, shape (..., n_keep, T_crop)
    lengths_new : updated 1-D array of trial lengths (n_keep,)
    """
    # ---------------------------------------------------------------
    # 1) decide which trials to keep
    thresh     = np.percentile(trial_lengths, drop_shortest_pct)
    keep_mask  = trial_lengths >= thresh                 # boolean, shape (n_trials,)
    lengths_new = trial_lengths[keep_mask]

    if keep_mask.sum() == 0:
        raise ValueError("Nothing left after dropping – lower your drop_shortest_pct")

    # ---------------------------------------------------------------
    # 2) keep only the selected trials
    arr_kept = arr[..., keep_mask, :]                    # (..., n_keep, T_max)

    # ---------------------------------------------------------------
    # 3) truncate to the minimum length of the kept trials
    T_crop   = lengths_new.min()
    arr_kept = arr_kept[..., :T_crop]                    # (..., n_keep, T_crop)

    return arr_kept, T_crop


def plot_state_timecourses(
        decoding,            # shape (n_states, n_trials, T)
                            #  → mean over axis=1 before plotting
        time_step   = 1.0,   #   seconds between samples; change if your bin width differs
        sem_band    = True,  #   draw ±SEM shading?
        title       = "",    #   panel title
        ylabel      = "decoding error",
        figsize     = (6, 4)):
    """
    Draw one line per continuous state, averaged over trials.

    - `decoding`   : ndarray, (n_states, n_trials, T)
    - `time_step`  : abscissa spacing (e.g. 0.02 → 50 Hz bins)
    """
    n_states, n_trials, T = decoding.shape
    t_axis = np.arange(T) * time_step

    means = np.nanmean(decoding, axis=1)               # (n_states, T)
    sems  = np.nanstd(decoding, axis=1, ddof=1) / np.sqrt(n_trials)  # (n_states, T)

    fig, ax = plt.subplots(figsize=figsize)

    for k in range(n_states):
        ax.plot(t_axis, means[k], label=f"state {k+1}")
        if sem_band:
            ax.fill_between(t_axis,
                            means[k] - sems[k],
                            means[k] + sems[k],
                            alpha=0.3)                 # default colour's alpha only

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_per_time_decoding_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    label_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    truncate_percentile=10,
    visual_delay_time=0):

    print('Plotting decoding results...')

    rSLDS_same_speed_decoding_name = '_'.join(map(str, [x for x in [
        'decoding_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_decoding_name = '_'.join(map(str, [x for x in [
        'decoding_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        'joint',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_decoding_slow_all = []
    rSLDS_same_speed_decoding_fast_all = []
    rSLDS_cross_speed_decoding_slow_all = []
    rSLDS_cross_speed_decoding_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_decoding = np.load(os.path.join(session_results_dir, rSLDS_same_speed_decoding_name + '.npz'))
        rSLDS_same_speed_decoding_slow = rSLDS_same_speed_decoding['decoding_errors_slow_test_per_time']
        rSLDS_same_speed_decoding_fast = rSLDS_same_speed_decoding['decoding_errors_fast_test_per_time']
        trial_lengths_slow_same_speed  = rSLDS_same_speed_decoding['trial_lengths_slow']
        trial_lengths_fast_same_speed  = rSLDS_same_speed_decoding['trial_lengths_fast']

        rSLDS_cross_speed_decoding = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_decoding_name + '.npz'))
        rSLDS_cross_speed_decoding_slow = rSLDS_cross_speed_decoding['decoding_errors_slow_test_per_time']
        rSLDS_cross_speed_decoding_fast = rSLDS_cross_speed_decoding['decoding_errors_fast_test_per_time']
        trial_lengths_slow_cross_speed  = rSLDS_cross_speed_decoding['trial_lengths_slow']
        trial_lengths_fast_cross_speed  = rSLDS_cross_speed_decoding['trial_lengths_fast']

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Take the first discrete state (LDS) and last n_iters
        rSLDS_same_speed_decoding_slow = rSLDS_same_speed_decoding_slow[:, :, 0, -1]
        rSLDS_same_speed_decoding_fast = rSLDS_same_speed_decoding_fast[:, :, 0, -1]
        rSLDS_cross_speed_decoding_slow = rSLDS_cross_speed_decoding_slow[:, :, 0, -1]
        rSLDS_cross_speed_decoding_fast = rSLDS_cross_speed_decoding_fast[:, :, 0, -1]

        # Truncate the decoding results with specified percentile
        rSLDS_same_speed_decoding_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_decoding_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_decoding_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_decoding_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_decoding_slow, _ = drop_and_truncate(rSLDS_cross_speed_decoding_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_decoding_fast, _ = drop_and_truncate(rSLDS_cross_speed_decoding_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_decoding_slow = np.mean(rSLDS_same_speed_decoding_slow, axis=0)
        rSLDS_same_speed_decoding_fast = np.mean(rSLDS_same_speed_decoding_fast, axis=0)
        rSLDS_cross_speed_decoding_slow = np.mean(rSLDS_cross_speed_decoding_slow, axis=0)
        rSLDS_cross_speed_decoding_fast = np.mean(rSLDS_cross_speed_decoding_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_decoding_slow_all.append(rSLDS_same_speed_decoding_slow)
        rSLDS_same_speed_decoding_fast_all.append(rSLDS_same_speed_decoding_fast)
        rSLDS_cross_speed_decoding_slow_all.append(rSLDS_cross_speed_decoding_slow)
        rSLDS_cross_speed_decoding_fast_all.append(rSLDS_cross_speed_decoding_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_decoding_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_decoding_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_decoding_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_decoding_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_decoding_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_decoding_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_decoding_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_decoding_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_decoding_slow_all[i] = rSLDS_same_speed_decoding_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_decoding_fast_all[i] = rSLDS_same_speed_decoding_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_decoding_slow_all[i] = rSLDS_cross_speed_decoding_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_decoding_fast_all[i] = rSLDS_cross_speed_decoding_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_decoding_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_decoding_slow_all[i]
        rSLDS_same_speed_decoding_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_decoding_fast_all[i]
        rSLDS_cross_speed_decoding_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_decoding_slow_all[i]
        rSLDS_cross_speed_decoding_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_decoding_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    decoding_results = {
        trial_filters[0]:
            {
                'LDS_mean_same_speed': rSLDS_same_speed_decoding_fast_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_fast_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_decoding_fast_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_fast_all[0],
            },
        trial_filters[1]:
            {
                'LDS_mean_same_speed': rSLDS_same_speed_decoding_slow_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_slow_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_decoding_slow_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_decoding_slow_all[0],
            }
    }

    # ipdb.set_trace()
    
    vmin = 0
    vmax = 90
    
    ## Plot results

    fig1 = plt.figure(figsize=(24, 12))
    ax0 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig1.add_subplot(1, 2, 2, projection='3d')
    axs = [ax0, ax1]

    for i_tf, trial_filter in enumerate(trial_filters):

        # Plot lines and fill under them
        for i_cs, n_states in enumerate(ns_states):

            color_list = ['blue', 'red']

            zs_list = [
                np.mean(decoding_results[trial_filter]['LDS_mean_same_speed'], axis=1)[i_cs],
                np.mean(decoding_results[trial_filter]['LDS_mean_cross_speed'], axis=1)[i_cs]
            ]

            for color, zs in zip(color_list, zs_list):

                trial_length = zs.shape[-1]

                xs = np.arange(trial_length)
                ys = np.full(trial_length, n_states)

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=color, 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=color, 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

        if trial_filter in ['slow', 'far']:
            xtick_step = 5
        else:
            xtick_step = 1

        axs[i_tf].set_xticks(xs[::xtick_step])
        axs[i_tf].set_xticklabels(np.round(xs[::xtick_step] * config.time_step + visual_delay_time, 2))
        axs[i_tf].set_yticks(ns_states)


        # Set labels and title
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel('# of Latent States', fontsize=label_fontsize)
        axs[i_tf].set_zlabel('Absolute Angle Error (degrees)', fontsize=label_fontsize)

        # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
        # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
        axs[i_tf].set_zlim(vmin, vmax)

        # axs[i_tf].set_yticklabels(['PCA'] + ns_discrete_states)

        # Adjust viewing angle
        # axs[i_tf].view_init(elev=10, azim=-75)
        axs[i_tf].view_init(elev=20, azim=-40)

        # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

        ## Remove background panes
        axs[i_tf].xaxis.pane.fill = False
        axs[i_tf].yaxis.pane.fill = False
        axs[i_tf].zaxis.pane.fill = False

        ## Set grid colors
        # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

        # ## Create tick labels: label only at even indices, else empty string
        # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

        # ## Set tick labels
        # axs[i_tf].set_xticklabels(tick_labels)


    ## Add custom legend
    # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

    # custom_lines = [
    #     Line2D([0], [0], color='blue',                   lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

    # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

    # ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     title='# of Discrete States',
    #     loc='upper center', 
    #     bbox_to_anchor=(-0.2, -0.15), 
    #     fancybox=False, 
    #     shadow=False, 
    #     ncol=9)

    # plt.show()

    ## Write image
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_decoding',
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        label_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig1.savefig(save_path, dpi=600, transparent=True)
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig1)


def plot_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    inference_type='forecast',
    form='heatmap'):

    print('Plotting inference results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))
    
    inference_baseline_name = '_'.join(map(str, [x for x in [
        'inference_baseline',
        unit_filter,
        data_format] if x is not None]))

    rSLDS_r2_forecast_slow_all  = []
    rSLDS_r2_forecast_fast_all  = []
    # r2_forecast_baseline_slow_all = []
    # r2_forecast_baseline_fast_all = []

    inference_name = 'r2_' + inference_type

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_inference = np.load(os.path.join(session_results_dir, rSLDS_inference_name + '.npz'))
        rSLDS_r2_forecast_slow = rSLDS_inference[inference_name + '_slow']
        rSLDS_r2_forecast_fast = rSLDS_inference[inference_name + '_fast']
        # inference_baseline = np.load(os.path.join(session_results_dir, inference_baseline_name + '.npz'))
        # r2_forecast_baseline_slow = inference_baseline['r2_forecast_baseline_slow']
        # r2_forecast_baseline_fast = inference_baseline['r2_forecast_baseline_fast']
        ipdb.set_trace()

        ## Take the last n_iters, then add to all sessions
        rSLDS_r2_forecast_slow = rSLDS_r2_forecast_slow[..., -1]
        rSLDS_r2_forecast_fast = rSLDS_r2_forecast_fast[..., -1]
        # r2_forecast_baseline_slow = r2_forecast_baseline_slow
        # r2_forecast_baseline_fast = r2_forecast_baseline_fast

        rSLDS_r2_forecast_slow_all.append(rSLDS_r2_forecast_slow)
        rSLDS_r2_forecast_fast_all.append(rSLDS_r2_forecast_fast)
        # r2_forecast_baseline_slow_all.append(r2_forecast_baseline_slow)
        # r2_forecast_baseline_fast_all.append(r2_forecast_baseline_fast)

    ## Stack all sessions
    rSLDS_r2_forecast_slow_all  = np.stack(rSLDS_r2_forecast_slow_all,  axis=0)
    rSLDS_r2_forecast_fast_all  = np.stack(rSLDS_r2_forecast_fast_all,  axis=0)
    # r2_forecast_baseline_slow_all = np.stack(r2_forecast_baseline_slow_all, axis=0)
    # r2_forecast_baseline_fast_all = np.stack(r2_forecast_baseline_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_r2_forecast_slow_all_mean = np.mean(rSLDS_r2_forecast_slow_all, axis=(0, 2, 3))
    rSLDS_r2_forecast_fast_all_mean = np.mean(rSLDS_r2_forecast_fast_all, axis=(0, 2, 3))
    # r2_forecast_baseline_slow_all_mean = np.mean(r2_forecast_baseline_slow_all, axis=(0, 2))
    # r2_forecast_baseline_fast_all_mean = np.mean(r2_forecast_baseline_fast_all, axis=(0, 2))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_r2_forecast_slow_all_se = np.std(rSLDS_r2_forecast_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_r2_forecast_fast_all_se = np.std(rSLDS_r2_forecast_fast_all, axis=(0, 2, 3)) / correction_factor
    # r2_forecast_baseline_slow_all_se = np.std(r2_forecast_baseline_slow_all, axis=(0, 2)) / correction_factor
    # r2_forecast_baseline_fast_all_se = np.std(r2_forecast_baseline_fast_all, axis=(0, 2)) / correction_factor


    inference_results = {
        trial_filters[1]:
            {
                'mean': rSLDS_r2_forecast_slow_all_mean,
                'se': rSLDS_r2_forecast_slow_all_se,
            },
        trial_filters[0]:
            {
                'mean': rSLDS_r2_forecast_fast_all_mean,
                'se': rSLDS_r2_forecast_fast_all_se,
            }
    }

    vmin = 0.5
    vmax = 1

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                inference_results[trial_filter]['mean'][train_or_test_id, ...].T,
                ax=axs[i_tf],
                vmin=vmin,
                vmax=vmax,
                cmap='rocket',
                cbar=False,          # ← suppress the color-bar
                # square=True,         # ← make the axes a true square
                xticklabels=False,   # ← drop x-axis tick labels
                yticklabels=False,   # ← drop y-axis tick labels
                linewidths=0,
                linecolor=None,
                rasterized=True,
            )
        
            # If you also want to hide the tick marks themselves:
            axs[i_tf].tick_params(left=False, bottom=False)
            axs[i_tf].set_box_aspect(1)

            # axs[i_tf].set_title('Variance explained (R²)', fontsize=5)
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_xticklabels(ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            # axs[i_tf].set_xlabel('Number of discrete states', fontsize=5)
            # axs[i_tf].set_ylabel('Number of continuous states', fontsize=5)
            axs[i_tf].invert_yaxis()

        fig.tight_layout()

    elif form == 'waterfall':

        fig = plt.figure(figsize=(50*mm, 25*mm))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            # Plot lines and fill under them
            for i_cs in range(len(ns_states)):

                xs = np.arange(len(ns_discrete_states))
                ys = np.full_like(ns_discrete_states, i_cs)
                zs = inference_results[trial_filter]['mean'][train_or_test_id, i_cs]

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=continuous_state_colors[i_cs], 
                    lw=0.25,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=continuous_state_colors[i_cs], 
                    alpha=0.1)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

            # Set labels and title
            # axs[i_tf].set_ylabel('Number of continuous states', fontsize=5, labelpad=-8)
            # axs[i_tf].set_xlabel('Number of discrete states', fontsize=5, labelpad=-8)
            # axs[i_tf].set_zlabel('Variance explained (R²)', fontsize=5, labelpad=-8)

            # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
            # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
            axs[i_tf].set_zlim(vmin, vmax)

            axs[i_tf].set_yticks(np.arange(len(ns_states)))
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_yticklabels([])
            axs[i_tf].set_xticks(np.arange(len(ns_discrete_states)))
            # axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_xticklabels([])

            axs[i_tf].set_zticklabels([])

            # Adjust viewing angle
            axs[i_tf].view_init(elev=20, azim=-40)

            # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

            ## Remove background panes
            axs[i_tf].xaxis.pane.fill = False
            axs[i_tf].yaxis.pane.fill = False
            axs[i_tf].zaxis.pane.fill = False
            axs[i_tf].grid(False) 

            ## Set grid colors
            # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

            # ## Create tick labels: label only at even indices, else empty string
            # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            # # ## Set tick labels
            # axs[i_tf].set_yticklabels(tick_labels)




    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'inference',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        inference_type,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    plt.close(fig)


def plot_per_time_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    inference_type='forecast',
    truncate_percentile=10,
    visual_delay_time=0,
    peak_time=None,
    discrete_state_idx=0,
    session_data_names=None,
    results_dir=None,
    vis_dir=None,
    window_config=None,
    time_offset=None,
    trial_filters=None,
    ns_states=None,
    random_states=None,
    n_folds=None,
    ns_discrete_states=None,
    ns_iters=None):
    """
    Parameters
    ----------
    discrete_state_idx : int
        Index into ns_discrete_states to select the discrete-state count.
        0 → LDS (K=1), 1 → rSLDS (K=2), etc.
    """

    # Resolve overridable config parameters
    if session_data_names is None:
        session_data_names = config.session_data_names
    if results_dir is None:
        results_dir = config.results_dir
    if vis_dir is None:
        vis_dir = config.vis_dir
    if window_config is None:
        window_config = config.window_config
    if time_offset is None:
        time_offset = config.time_offset
    if trial_filters is None:
        trial_filters = config.trial_filters
    if ns_states is None:
        ns_states = config.ns_states
    if random_states is None:
        random_states = config.random_states
    if n_folds is None:
        n_folds = config.n_folds
    if ns_discrete_states is None:
        ns_discrete_states = config.ns_discrete_states
    if ns_iters is None:
        ns_iters = config.ns_iters

    # npz keys always use fast/slow even for RadialGrid (near/far)
    _tf_data_key_map = {'near': 'fast', 'far': 'slow'}
    tf0_data = _tf_data_key_map.get(trial_filters[0], trial_filters[0])
    tf1_data = _tf_data_key_map.get(trial_filters[1], trial_filters[1])

    n_discrete = ns_discrete_states[discrete_state_idx]
    model_label = 'LDS' if n_discrete == 1 else f'rSLDS (K={n_discrete})'
    print(f'Plotting inference results for {model_label}...')

    rSLDS_same_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        # input_unit_filter,
        None,
        window_config,
        # time_offset,
        None,
        data_format,
        trial_filters,
        'cross_speed',
        # 'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_inference_slow_all = []
    rSLDS_same_speed_inference_fast_all = []
    rSLDS_cross_speed_inference_slow_all = []
    rSLDS_cross_speed_inference_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    inference_name = 'r2_' + inference_type

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_inference = np.load(os.path.join(session_results_dir, rSLDS_same_speed_inference_name + '.npz'))
        rSLDS_same_speed_inference_slow = rSLDS_same_speed_inference[inference_name + f'_{tf1_data}_test_per_time']
        rSLDS_same_speed_inference_fast = rSLDS_same_speed_inference[inference_name + f'_{tf0_data}_test_per_time']
        trial_lengths_slow_same_speed = rSLDS_same_speed_inference[f'trial_lengths_{tf1_data}'] - 1
        trial_lengths_fast_same_speed = rSLDS_same_speed_inference[f'trial_lengths_{tf0_data}'] - 1

        rSLDS_cross_speed_inference = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_inference_name + '.npz'))
        rSLDS_cross_speed_inference_slow = rSLDS_cross_speed_inference[inference_name + f'_{tf1_data}_test_per_time']
        rSLDS_cross_speed_inference_fast = rSLDS_cross_speed_inference[inference_name + f'_{tf0_data}_test_per_time']
        trial_lengths_slow_cross_speed = rSLDS_cross_speed_inference[f'trial_lengths_{tf1_data}'] - 1
        trial_lengths_fast_cross_speed = rSLDS_cross_speed_inference[f'trial_lengths_{tf0_data}'] - 1

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Select discrete state and last iteration
        # discrete_state_idx=0 → LDS (K=1), 1 → rSLDS (K=2), etc.
        rSLDS_same_speed_inference_slow = rSLDS_same_speed_inference_slow[:, :, discrete_state_idx, -1]
        rSLDS_same_speed_inference_fast = rSLDS_same_speed_inference_fast[:, :, discrete_state_idx, -1]
        rSLDS_cross_speed_inference_slow = rSLDS_cross_speed_inference_slow[:, :, discrete_state_idx, -1]
        rSLDS_cross_speed_inference_fast = rSLDS_cross_speed_inference_fast[:, :, discrete_state_idx, -1]

        # Truncate the inference results with specified percentile
        rSLDS_same_speed_inference_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_inference_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_inference_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_inference_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_inference_slow, _ = drop_and_truncate(rSLDS_cross_speed_inference_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_inference_fast, _ = drop_and_truncate(rSLDS_cross_speed_inference_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_inference_slow = np.mean(rSLDS_same_speed_inference_slow, axis=0)
        rSLDS_same_speed_inference_fast = np.mean(rSLDS_same_speed_inference_fast, axis=0)
        rSLDS_cross_speed_inference_slow = np.mean(rSLDS_cross_speed_inference_slow, axis=0)
        rSLDS_cross_speed_inference_fast = np.mean(rSLDS_cross_speed_inference_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_inference_slow_all.append(rSLDS_same_speed_inference_slow)
        rSLDS_same_speed_inference_fast_all.append(rSLDS_same_speed_inference_fast)
        rSLDS_cross_speed_inference_slow_all.append(rSLDS_cross_speed_inference_slow)
        rSLDS_cross_speed_inference_fast_all.append(rSLDS_cross_speed_inference_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_inference_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_inference_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_inference_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_inference_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_inference_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_inference_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_inference_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_inference_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_inference_slow_all[i] = rSLDS_same_speed_inference_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_inference_fast_all[i] = rSLDS_same_speed_inference_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_inference_slow_all[i] = rSLDS_cross_speed_inference_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_inference_fast_all[i] = rSLDS_cross_speed_inference_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_inference_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_inference_slow_all[i]
        rSLDS_same_speed_inference_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_inference_fast_all[i]
        rSLDS_cross_speed_inference_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_inference_slow_all[i]
        rSLDS_cross_speed_inference_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_inference_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    inference_results = {
        trial_filters[0]:
            {
                'same_speed': rSLDS_same_speed_inference_fast_all_,
                'cross_speed': rSLDS_cross_speed_inference_fast_all_,
            },
        trial_filters[1]:
            {
                'same_speed': rSLDS_same_speed_inference_slow_all_,
                'cross_speed': rSLDS_cross_speed_inference_slow_all_,
            }
    }
    
    vmin = 0.5
    vmax = 1

    ## ── Time axis setup ──────────────────────────────────────────────
    dt = config.time_step
    t0 = visual_delay_time

    # Build time vectors per condition (trial lengths may differ)
    time_vectors = {}
    t_max_all = 0
    for trial_filter in trial_filters:
        T_cond = inference_results[trial_filter]['same_speed'].shape[-1]
        time_vectors[trial_filter] = t0 + np.arange(T_cond) * dt
        t_max_all = max(t_max_all, time_vectors[trial_filter][-1])

    # Column width ratios proportional to trial duration (equalize time scale)
    durations = [time_vectors[tf][-1] - time_vectors[tf][0] for tf in trial_filters]
    width_ratios = [d / max(durations) for d in durations]

    ## ══════════════════════════════════════════════════════════════════
    ## Fig 1 — R^2 heatmaps: 2 rows x 2 cols
    ##          Row 0 = same-condition, Row 1 = cross-condition
    ## ══════════════════════════════════════════════════════════════════
    n_dims = len(ns_states)
    panel_labels = ['Same-condition', 'Cross-condition']
    data_keys = ['same_speed', 'cross_speed']

    fig1, axs1 = plt.subplots(
        2, len(trial_filters), figsize=(90 * mm, 80 * mm),
        sharey=True,
        gridspec_kw={'width_ratios': width_ratios})

    def build_tick_info(t_vec_local, step=0.25):
        """Build tick positions and labels for a given time vector.

        Parameters
        ----------
        t_vec_local : array
            Time vector for the condition.
        step : float
            Tick spacing in seconds after the first tick (t0).
            E.g. 0.25 -> t0, 0.25, 0.5, 0.75, ...
                 0.2  -> t0, 0.2, 0.4, 0.6, ...
        """
        t_last_local = t_vec_local[-1]
        first_nice = np.ceil(t0 / step) * step
        if first_nice - t0 < 0.01:
            first_nice += step
        nice_ticks = np.arange(first_nice, t_last_local + 1e-9, step)
        positions = np.concatenate([[t0], nice_ticks])
        def _fmt(t):
            """Format: integers as '1', otherwise up to 2 decimals with no trailing zeros."""
            if abs(t - round(t)) < 1e-9:
                return f'{t:.0f}'
            s = f'{t:.2f}'.rstrip('0')
            return s
        labels = [_fmt(t0)] + [_fmt(t) for t in nice_ticks]
        return positions, labels

    for i_key, key in enumerate(data_keys):
        for i_tf, trial_filter in enumerate(trial_filters):
            ax = axs1[i_key, i_tf]
            t_vec = time_vectors[trial_filter]
            cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)

            r2_matrix = np.mean(
                inference_results[trial_filter][key], axis=1)

            im = ax.pcolormesh(
                t_vec, ns_states, r2_matrix,
                vmin=vmin, vmax=vmax,
                cmap='rocket', shading='nearest', rasterized=True)

            tick_pos, tick_lab = build_tick_info(t_vec)
            ax.set_xticks(tick_pos)
            if i_key == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(tick_lab)

            ax.set_yticks(ns_states)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if peak_time is not None:
                ax.axvline(peak_time, color="black", linewidth=size_line_thin,
                           linestyle=":", alpha=alpha_line)

    # fig1.suptitle(model_label, fontsize=8)

    ## ── Standalone horizontal colorbar (no labels) ──
    cbar_fig, cbar_ax = plt.subplots(figsize=(20 * mm, 2 * mm))
    cbar_norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = cbar_fig.colorbar(
        plt.cm.ScalarMappable(norm=cbar_norm, cmap='rocket'),
        cax=cbar_ax,
        orientation='horizontal')
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    cbar_ax.set_frame_on(False)

    ## ══════════════════════════════════════════════════════════════════
    ## Fig 2 — Difference heatmap (same - cross), no significance overlay
    ## ══════════════════════════════════════════════════════════════════
    diff_max = 0.2
    div_cmap = sns.color_palette('vlag', as_cmap=True)

    fig2, axs2 = plt.subplots(
        1, 2, figsize=(90 * mm, 40 * mm),
        sharey=True,
        gridspec_kw={'width_ratios': width_ratios})

    for i_tf, trial_filter in enumerate(trial_filters):
        ax = axs2[i_tf]
        t_vec = time_vectors[trial_filter]
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)

        same = inference_results[trial_filter]['same_speed']
        cross = inference_results[trial_filter]['cross_speed']
        diff = same - cross  # positive => same-condition better

        # Mean difference over trials
        diff_mean = np.mean(diff, axis=1)

        im2 = ax.pcolormesh(
            t_vec, ns_states, diff_mean,
            vmin=-diff_max, vmax=diff_max,
            cmap=div_cmap, shading='nearest', rasterized=True)

        # ax.set_title(cond_label)
        tick_pos, tick_lab = build_tick_info(t_vec)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        # ax.set_xlabel('Time (s)')
        ax.set_yticks(ns_states)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if peak_time is not None:
            ax.axvline(peak_time, color="black", linewidth=size_line_thin,
                       linestyle=":", alpha=alpha_line)

    # axs2[0].set_ylabel('Number of latent states')

    ## ── Standalone horizontal colorbar for fig2 (no labels) ──
    cbar2_fig, cbar2_ax = plt.subplots(figsize=(20 * mm, 2 * mm))
    cbar2_norm = Normalize(vmin=-diff_max, vmax=diff_max)
    cbar2 = cbar2_fig.colorbar(
        plt.cm.ScalarMappable(norm=cbar2_norm, cmap=div_cmap),
        cax=cbar2_ax,
        orientation='horizontal')
    cbar2.set_ticks([])
    cbar2.outline.set_visible(False)
    cbar2_ax.set_frame_on(False)

    ## ══════════════════════════════════════════════════════════════════
    ## Save figures
    ## ══════════════════════════════════════════════════════════════════
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    discrete_label = f'K{n_discrete}'

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_inference',
        inference_type,
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha,
        discrete_label] if x is not None]))

    save_path1 = os.path.join(vis_dir, img_name + '_heatmap.pdf')
    save_path2 = os.path.join(vis_dir, img_name + '_diff_heatmap.pdf')

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(save_path1, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    fig2.savefig(save_path2, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    cbar_fig.savefig(os.path.join(vis_dir, img_name + '_colorbar.pdf'),
                     bbox_inches=None, transparent=True, dpi=600, format='pdf')
    cbar2_fig.savefig(os.path.join(vis_dir, img_name + '_diff_colorbar.pdf'),
                      bbox_inches=None, transparent=True, dpi=600, format='pdf')
    plt.close(fig1)
    plt.close(fig2)
    plt.close(cbar_fig)
    plt.close(cbar2_fig)


def plot_rslds_phase_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    n_discrete=2,
    inference_type='forecast',
    truncate_percentile=10,
    visual_delay_time=0.0):
    """
    rSLDS-specific analysis: compare same-vs-cross R^2 separately for
    transient and steady discrete-state phases.

    For each trial, the posterior discrete state sequence (from the .npz
    field ``discrete_states_{fast,slow}_test_per_time``) partitions time
    bins into phases.  We collect per-time R^2 values within each phase,
    then compare same-condition vs cross-condition performance by phase.
    """

    print(f'Plotting rSLDS phase inference (K={n_discrete})...')

    discrete_state_idx = ns_discrete_states.index(n_discrete)
    inference_name = 'r2_' + inference_type

    # ── Build file names for inference .npz ──
    same_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    cross_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        None,
        window_config,
        None,
        data_format,
        trial_filters,
        'cross_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    # ── Per-phase R^2 collection across sessions ──
    # For each (trial_filter, same/cross, n_continuous_states, phase):
    #   collect lists of scalar R^2 values (one per trial)
    # Store per-time-point delta R^2 (same - cross) by phase
    phase_r2 = {
        tf: {
            'same_minus_cross': {ns: {phase: [] for phase in range(n_discrete)}
                                 for ns in ns_states}
        }
        for tf in trial_filters
    }

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        # Load inference .npz arrays
        same_inf = np.load(os.path.join(
            session_results_dir, same_speed_inference_name + '.npz'))
        cross_inf = np.load(os.path.join(
            session_results_dir, cross_speed_inference_name + '.npz'))

        for i_tf, trial_filter in enumerate(trial_filters):
            tf_key = trial_filter

            # R^2 per time: shape (n_rs, n_ns_states, n_ns_discrete, n_ns_iters, n_trials, T_r2)
            same_r2_per_time = same_inf[inference_name + f'_{tf_key}_test_per_time']
            cross_r2_per_time = cross_inf[inference_name + f'_{tf_key}_test_per_time']

            # Discrete states per time: shape (n_rs, n_ns_states, n_ns_discrete, n_ns_iters, n_trials, T_ds)
            # Values: integer 0..K-1, padded with -1
            ds_per_time = same_inf[f'discrete_states_{tf_key}_test_per_time']

            trial_lengths_full = same_inf[f'trial_lengths_{tf_key}']
            trial_lengths = trial_lengths_full - 1  # forecast R^2 is T-1

            # Select discrete state idx and last iteration, average over random states
            # -> shape (n_ns_states, n_trials, T)
            same_r2 = np.mean(same_r2_per_time[:, :, discrete_state_idx, -1], axis=0)
            cross_r2 = np.mean(cross_r2_per_time[:, :, discrete_state_idx, -1], axis=0)

            # Discrete states: use first random state (deterministic given the model)
            # -> shape (n_ns_states, n_trials, T_ds)
            ds_all = ds_per_time[0, :, discrete_state_idx, -1]

            n_trials = same_r2.shape[1]

            for i_ns, n_continuous_states in enumerate(ns_states):

                # ── Reorder discrete states so state 0 = earliest (transient) ──
                # Collect valid discrete state sequences for this latent dim
                ds_sequences = []
                for i_trial in range(n_trials):
                    tl_full = int(trial_lengths_full[i_trial])
                    ds_trial_full = ds_all[i_ns, i_trial, :tl_full]
                    if not np.all(ds_trial_full == -1):
                        ds_sequences.append(ds_trial_full)

                if len(ds_sequences) == 0:
                    continue

                # Reorder: state that appears earliest -> 0 (transient)
                ds_sequences_reordered, old_to_new, _ = utils_vis.reorder_discrete_states(
                    ds_sequences, n_discrete)

                # Build a lookup from trial index to reordered sequence
                reordered_idx = 0
                for i_trial in range(n_trials):
                    tl_full = int(trial_lengths_full[i_trial])
                    ds_trial_full = ds_all[i_ns, i_trial, :tl_full]
                    if np.all(ds_trial_full == -1):
                        continue

                    tl = int(trial_lengths[i_trial])
                    tl_r2 = min(tl, same_r2.shape[-1])

                    same_trial_r2 = same_r2[i_ns, i_trial, :tl_r2]
                    cross_trial_r2 = cross_r2[i_ns, i_trial, :tl_r2]

                    # Align: forecast at t uses ds[t] to predict x[t+1],
                    # so R^2[t] corresponds to ds[t]. R^2 has length T-1.
                    ds_reordered = ds_sequences_reordered[reordered_idx]
                    ds_trial = ds_reordered[:tl_r2]
                    reordered_idx += 1

                    # Store individual time-point differences (same - cross)
                    diff_trial_r2 = same_trial_r2 - cross_trial_r2

                    for phase in range(n_discrete):
                        mask = ds_trial == phase
                        if mask.sum() > 0:
                            phase_r2[trial_filter]['same_minus_cross'][n_continuous_states][phase].extend(
                                diff_trial_r2[mask].tolist())

    # ── Plotting ──
    # Box plots of per-time-point delta R^2 (same - cross), split by phase
    # Key question: is the gap larger in the steady phase?
    phase_names = {0: 'Transient', 1: 'Steady'}

    # Use task-specific discrete state colors (dark = transient, light = steady)
    task_name = session_data_names[0].split('_')[-1]
    from vis_config import discrete_state_colors as dsc

    fig, axs = plt.subplots(
        1, len(trial_filters),
        figsize=(120 * mm, 55 * mm),
        sharey=True)
    if len(trial_filters) == 1:
        axs = [axs]

    print(f'\n{"="*70}')
    print(f'rSLDS phase analysis (K={n_discrete}): transient vs steady')
    print(f'{"="*70}')

    for i_tf, trial_filter in enumerate(trial_filters):
        ax = axs[i_tf]
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)

        # Phase colors: state 0 (transient) = dark, state 1 (steady) = light
        tf_colors = dsc[task_name][trial_filter]
        phase_palette = {
            'Transient': tf_colors[0],  # dark
            'Steady': tf_colors[1],     # light
        }

        # Build DataFrame: each row is one time-point delta R^2
        rows = []
        for ns in ns_states:
            for phase in range(n_discrete):
                vals = phase_r2[trial_filter]['same_minus_cross'][ns][phase]
                for v in vals:
                    rows.append({
                        'Latent dims': ns,
                        'Phase': phase_names.get(phase, f'State {phase}'),
                        'delta_R2': v,
                    })

        if len(rows) == 0:
            ax.set_title(cond_label + '\n(no data)')
            continue

        df = pd.DataFrame(rows)

        sns.boxplot(
            data=df, x='Latent dims', y='delta_R2',
            hue='Phase', ax=ax,
            fliersize=0.5, linewidth=0.5,
            palette=phase_palette,
            showfliers=False)

        ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
        ax.set_title(cond_label)
        ax.set_xlabel('Number of latent states')
        ax.spines[['right', 'top']].set_visible(False)

        if i_tf > 0:
            ax.get_legend().remove()

        # ── Statistics: MWU + rank-biserial r for each latent dim ──
        print(f'\n{cond_label}:')
        print(f'{"n_states":>10} {"U":>10} {"p":>12} {"r_rb":>8} '
              f'{"n_trans":>8} {"n_steady":>8} {"med_trans":>10} {"med_steady":>10}')
        print('-' * 90)

        for ns in ns_states:
            trans_vals = np.array(phase_r2[trial_filter]['same_minus_cross'][ns][0])
            steady_vals = np.array(phase_r2[trial_filter]['same_minus_cross'][ns][1])

            if len(trans_vals) < 2 or len(steady_vals) < 2:
                print(f'{ns:>10}   insufficient data')
                continue

            stat, p_val = mannwhitneyu(trans_vals, steady_vals, alternative='two-sided')
            n1, n2 = len(trans_vals), len(steady_vals)
            r_rb = 1 - 2 * stat / (n1 * n2)

            print(f'{ns:>10} {stat:>10.0f} {p_val:>12.3g} {r_rb:>8.3f} '
                  f'{n1:>8} {n2:>8} {np.median(trans_vals):>10.4f} {np.median(steady_vals):>10.4f}')

    axs[0].set_ylabel(r'$\Delta$R$^2$ (same $-$ cross)')
    fig.tight_layout()

    # ── Save ──
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'phase_inference',
        inference_type,
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha,
        f'K{n_discrete}'] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '_phase_delta_r2.pdf')
    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    plt.close(fig)

    ## ══════════════════════════════════════════════════════════════════
    ## Fig 4 — Pooled across latent dims: 4 boxes total
    ##          (ballistic transient, ballistic steady,
    ##           sustained transient, sustained steady)
    ## ══════════════════════════════════════════════════════════════════

    # Collect pooled delta R^2 per (trial_filter, phase)
    pooled = {}
    for trial_filter in trial_filters:
        pooled[trial_filter] = {phase: [] for phase in range(n_discrete)}
        for ns in ns_states:
            for phase in range(n_discrete):
                pooled[trial_filter][phase].extend(
                    phase_r2[trial_filter]['same_minus_cross'][ns][phase])

    # Build DataFrame
    rows_pooled = []
    for trial_filter in trial_filters:
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)
        for phase in range(n_discrete):
            phase_label = phase_names.get(phase, f'State {phase}')
            for v in pooled[trial_filter][phase]:
                rows_pooled.append({
                    'Condition': cond_label,
                    'Phase': phase_label,
                    'delta_R2': v,
                })

    df_pooled = pd.DataFrame(rows_pooled)

    # Build palette keyed by (Condition, Phase)
    pooled_palette = {}
    for trial_filter in trial_filters:
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)
        tf_colors = dsc[task_name][trial_filter]
        pooled_palette[(cond_label, 'Transient')] = tf_colors[0]
        pooled_palette[(cond_label, 'Steady')]     = tf_colors[1]

    fig4, ax4 = plt.subplots(figsize=(55 * mm, 55 * mm))

    sns.boxplot(
        data=df_pooled, x='Condition', y='delta_R2',
        hue='Phase', ax=ax4,
        fliersize=0.5, linewidth=0.5,
        palette={
            'Transient': dsc[task_name][trial_filters[0]][0],
            'Steady':    dsc[task_name][trial_filters[0]][1],
        },
        showfliers=False)

    ax4.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
    ax4.set_xlabel('')
    ax4.set_ylabel(r'$\Delta$R$^2$ (same $-$ cross)')
    ax4.spines[['right', 'top']].set_visible(False)

    # ── Significance annotations ──
    def _sig_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return 'n.s.'

    print(f'\n{"="*70}')
    print(f'Pooled phase analysis (K={n_discrete}): transient vs steady')
    print(f'{"="*70}')

    y_max = df_pooled['delta_R2'].quantile(0.98)
    bracket_dy = (df_pooled['delta_R2'].quantile(0.98)
                  - df_pooled['delta_R2'].quantile(0.02)) * 0.06

    for i_tf, trial_filter in enumerate(trial_filters):
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)
        trans_vals = np.array(pooled[trial_filter][0])
        steady_vals = np.array(pooled[trial_filter][1])

        stat, p_val = mannwhitneyu(trans_vals, steady_vals, alternative='two-sided')
        n1, n2 = len(trans_vals), len(steady_vals)
        r_rb = 1 - 2 * stat / (n1 * n2)

        print(f'{cond_label}: U={stat:.0f}, p={p_val:.3g}, r={r_rb:.3f}, '
              f'n_trans={n1}, n_steady={n2}, '
              f'med_trans={np.median(trans_vals):.4f}, med_steady={np.median(steady_vals):.4f}')

        # Bracket positions: two boxes per condition group
        x_left  = i_tf - 0.2
        x_right = i_tf + 0.2
        y_bar   = y_max + bracket_dy * (1 + i_tf * 0.5)

        ax4.plot([x_left, x_left, x_right, x_right],
                 [y_bar - bracket_dy * 0.3, y_bar, y_bar, y_bar - bracket_dy * 0.3],
                 lw=0.8, color='black')
        stars = _sig_stars(p_val)
        ax4.text((x_left + x_right) / 2, y_bar + bracket_dy * 0.15,
                 f'r={r_rb:.2f}\n{stars}',
                 ha='center', va='bottom', fontsize=6)

    fig4.tight_layout()

    save_path4 = os.path.join(vis_dir, img_name + '_phase_delta_r2_pooled.pdf')
    fig4.savefig(save_path4, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    plt.close(fig4)


def plot_lds_phase_inference_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    inference_type='forecast',
    truncate_percentile=10,
    visual_delay_time=0.0,
    peak_time=0.3,
    session_data_names=None,
    results_dir=None,
    vis_dir=None,
    window_config=None,
    time_offset=None,
    trial_filters=None,
    ns_states=None,
    random_states=None,
    n_folds=None,
    ns_discrete_states=None,
    ns_iters=None,
    plot_pooled=True):
    """
    LDS analysis: compare same-vs-cross R^2 for transient vs steady phases,
    partitioned by a fixed peak_time boundary (not discrete states).

    Transient = time bins before peak_time (relative to trial start).
    Steady    = time bins at or after peak_time.

    Uses the LDS model (K=1, discrete_state_idx=0).

    Parameters
    ----------
    visual_delay_time : float
        Time (s) of the first bin relative to movement onset.
    peak_time : float
        Time (s) that separates transient from steady phase.
    """

    # Resolve overridable config parameters
    if session_data_names is None:
        session_data_names = config.session_data_names
    if results_dir is None:
        results_dir = config.results_dir
    if vis_dir is None:
        vis_dir = config.vis_dir
    if window_config is None:
        window_config = config.window_config
    if time_offset is None:
        time_offset = config.time_offset
    if trial_filters is None:
        trial_filters = config.trial_filters
    if ns_states is None:
        ns_states = config.ns_states
    if random_states is None:
        random_states = config.random_states
    if n_folds is None:
        n_folds = config.n_folds
    if ns_discrete_states is None:
        ns_discrete_states = config.ns_discrete_states
    if ns_iters is None:
        ns_iters = config.ns_iters

    dt = config.time_step
    # Index of the first bin at or after peak_time
    peak_bin = int(np.ceil((peak_time - visual_delay_time) / dt))

    print(f'Plotting LDS phase inference (peak_time={peak_time}s, '
          f'peak_bin={peak_bin}, dt={dt})...')

    discrete_state_idx = 0  # LDS (K=1)
    inference_name = 'r2_' + inference_type

    # ── Build file names for inference .npz ──
    same_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    cross_speed_inference_name = '_'.join(map(str, [x for x in [
        'inference',
        unit_filter,
        None,
        window_config,
        None,
        data_format,
        trial_filters,
        'cross_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    # ── Per-phase R^2 collection across sessions ──
    phase_r2 = {
        tf: {
            'same_minus_cross': {ns: {phase: [] for phase in range(2)}
                                 for ns in ns_states}
        }
        for tf in trial_filters
    }

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        same_inf = np.load(os.path.join(
            session_results_dir, same_speed_inference_name + '.npz'))
        cross_inf = np.load(os.path.join(
            session_results_dir, cross_speed_inference_name + '.npz'))

        # npz keys always use fast/slow even for RadialGrid (near/far)
        _tf_data_key_map = {'near': 'fast', 'far': 'slow'}

        for i_tf, trial_filter in enumerate(trial_filters):
            tf_key = _tf_data_key_map.get(trial_filter, trial_filter)

            same_r2_per_time = same_inf[inference_name + f'_{tf_key}_test_per_time']
            cross_r2_per_time = cross_inf[inference_name + f'_{tf_key}_test_per_time']

            trial_lengths_full = same_inf[f'trial_lengths_{tf_key}']
            trial_lengths = trial_lengths_full - 1  # forecast R^2 is T-1

            # Select LDS (K=1) and last iteration, average over random states
            # -> shape (n_ns_states, n_trials, T)
            same_r2 = np.mean(same_r2_per_time[:, :, discrete_state_idx, -1], axis=0)
            cross_r2 = np.mean(cross_r2_per_time[:, :, discrete_state_idx, -1], axis=0)

            n_trials = same_r2.shape[1]

            for i_ns, n_continuous_states in enumerate(ns_states):
                for i_trial in range(n_trials):
                    tl = int(trial_lengths[i_trial])
                    tl_r2 = min(tl, same_r2.shape[-1])

                    same_trial_r2 = same_r2[i_ns, i_trial, :tl_r2]
                    cross_trial_r2 = cross_r2[i_ns, i_trial, :tl_r2]
                    diff_trial_r2 = same_trial_r2 - cross_trial_r2

                    # Partition by peak_bin
                    bin_cut = min(peak_bin, tl_r2)
                    if bin_cut > 0:
                        phase_r2[trial_filter]['same_minus_cross'][n_continuous_states][0].extend(
                            diff_trial_r2[:bin_cut].tolist())
                    if bin_cut < tl_r2:
                        phase_r2[trial_filter]['same_minus_cross'][n_continuous_states][1].extend(
                            diff_trial_r2[bin_cut:].tolist())

    # ── Helper: style a matplotlib boxplot like crossnobis_robustness_check ──
    def _style_bp(bp, color):
        bp['boxes'][0].set(facecolor=color, linewidth=0, alpha=0.8)
        for line in bp['whiskers'] + bp['caps']:
            line.set(color='black', linewidth=0.25)
        for line in bp['medians']:
            line.set(color='black', linewidth=0.5)

    def _sig_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return 'n.s.'

    # ── Plotting (Fig 3: per latent dim) ──
    task_name = session_data_names[0].split('_')[-1]
    from vis_config import discrete_state_colors as dsc

    n_ns = len(ns_states)
    fig, axs = plt.subplots(
        1, len(trial_filters),
        figsize=(180 * mm, 40 * mm),
        sharey=True)
    if len(trial_filters) == 1:
        axs = [axs]

    print(f'\n{"="*70}')
    print(f'LDS phase analysis (peak_time={peak_time}s): transient vs steady')
    print(f'{"="*70}')

    box_width = 0.35

    for i_tf, trial_filter in enumerate(trial_filters):
        ax = axs[i_tf]
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)

        tf_colors = dsc[task_name][trial_filter]
        color_trans = tf_colors[0]
        color_steady = tf_colors[1]

        x_positions = np.arange(n_ns)

        print(f'\n{cond_label}:')
        print(f'{"n_states":>10} {"U":>10} {"p":>12} {"r_rb":>8} '
              f'{"n_trans":>8} {"n_steady":>8} {"med_trans":>10} {"med_steady":>10}')
        print('-' * 90)

        for i_ns, ns in enumerate(ns_states):
            trans_vals = phase_r2[trial_filter]['same_minus_cross'][ns][0]
            steady_vals = phase_r2[trial_filter]['same_minus_cross'][ns][1]

            if len(trans_vals) > 0:
                bp_t = ax.boxplot(
                    trans_vals,
                    positions=[x_positions[i_ns] - box_width / 2],
                    widths=box_width * 0.8,
                    whis=[0, 100], showfliers=False,
                    patch_artist=True, manage_ticks=False)
                _style_bp(bp_t, color_trans)

            if len(steady_vals) > 0:
                bp_s = ax.boxplot(
                    steady_vals,
                    positions=[x_positions[i_ns] + box_width / 2],
                    widths=box_width * 0.8,
                    whis=[0, 100], showfliers=False,
                    patch_artist=True, manage_ticks=False)
                _style_bp(bp_s, color_steady)

            trans_arr = np.array(trans_vals)
            steady_arr = np.array(steady_vals)
            if len(trans_arr) < 2 or len(steady_arr) < 2:
                print(f'{ns:>10}   insufficient data')
                continue

            stat, p_val = mannwhitneyu(trans_arr, steady_arr, alternative='two-sided')
            n1, n2 = len(trans_arr), len(steady_arr)
            r_rb = 1 - 2 * stat / (n1 * n2)

            print(f'{ns:>10} {stat:>10.0f} {p_val:>12.3g} {r_rb:>8.3f} '
                  f'{n1:>8} {n2:>8} {np.median(trans_arr):>10.4f} {np.median(steady_arr):>10.4f}')

        ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
        # ax.set_title(cond_label)
        # ax.set_xlabel('Number of latent states')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ns_states)
        ax.spines[['right', 'top']].set_visible(False)

    # axs[0].set_ylabel(r'$\Delta$R$^2$ (same $-$ cross)')
    fig.tight_layout()

    # ── Save fig 3 ──
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'phase_inference_LDS',
        inference_type,
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha,
        f'peak{peak_time}'] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '_phase_delta_r2.pdf')
    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    plt.close(fig)

    if not plot_pooled:
        return phase_r2

    ## ══════════════════════════════════════════════════════════════════
    ## Fig 4 — Pooled across latent dims: 4 boxes total
    ## ══════════════════════════════════════════════════════════════════

    pooled = {}
    for trial_filter in trial_filters:
        pooled[trial_filter] = {phase: [] for phase in range(2)}
        for ns in ns_states:
            for phase in range(2):
                pooled[trial_filter][phase].extend(
                    phase_r2[trial_filter]['same_minus_cross'][ns][phase])

    fig4, ax4 = plt.subplots(figsize=(90 * mm, 40 * mm))

    box_width_pooled = 0.3

    for i_tf, trial_filter in enumerate(trial_filters):
        tf_colors = dsc[task_name][trial_filter]
        trans_vals = pooled[trial_filter][0]
        steady_vals = pooled[trial_filter][1]

        if len(trans_vals) > 0:
            bp_t = ax4.boxplot(
                trans_vals,
                positions=[i_tf - box_width_pooled / 2],
                widths=box_width_pooled * 0.8,
                whis=[0, 100], showfliers=False,
                patch_artist=True, manage_ticks=False)
            _style_bp(bp_t, tf_colors[0])

        if len(steady_vals) > 0:
            bp_s = ax4.boxplot(
                steady_vals,
                positions=[i_tf + box_width_pooled / 2],
                widths=box_width_pooled * 0.8,
                whis=[0, 100], showfliers=False,
                patch_artist=True, manage_ticks=False)
            _style_bp(bp_s, tf_colors[1])

    ax4.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
    ax4.set_xticks(range(len(trial_filters)))
    ax4.set_xticklabels([trial_filter_name_conversion.get(tf, tf)
                         for tf in trial_filters])
    ax4.set_xlabel('')
    ax4.set_ylabel(r'$\Delta$R$^2$ (same $-$ cross)')
    ax4.spines[['right', 'top']].set_visible(False)

    # ── Significance annotations ──
    print(f'\n{"="*70}')
    print(f'Pooled LDS phase analysis (peak_time={peak_time}s): transient vs steady')
    print(f'{"="*70}')

    y_max = ax4.get_ylim()[1]
    tip_h = (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.015

    for i_tf, trial_filter in enumerate(trial_filters):
        cond_label = trial_filter_name_conversion.get(trial_filter, trial_filter)
        trans_arr = np.array(pooled[trial_filter][0])
        steady_arr = np.array(pooled[trial_filter][1])

        stat, p_val = mannwhitneyu(trans_arr, steady_arr, alternative='two-sided')
        n1, n2 = len(trans_arr), len(steady_arr)
        r_rb = 1 - 2 * stat / (n1 * n2)

        print(f'{cond_label}: U={stat:.0f}, p={p_val:.3g}, r={r_rb:.3f}, '
              f'n_trans={n1}, n_steady={n2}, '
              f'med_trans={np.median(trans_arr):.4f}, med_steady={np.median(steady_arr):.4f}')

        x_left  = i_tf - box_width_pooled / 2
        x_right = i_tf + box_width_pooled / 2
        bar_top = max(np.max(trans_arr), np.max(steady_arr))
        bracket_h = bar_top + (y_max - ax4.get_ylim()[0]) * 0.05

        ax4.plot([x_left, x_left, x_right, x_right],
                 [bracket_h - tip_h, bracket_h, bracket_h, bracket_h - tip_h],
                 color='black', linewidth=0.5, clip_on=False)

        stars = _sig_stars(p_val)
        if stars == 'n.s.':
            label_text = 'n.s.'
        else:
            label_text = f'r={r_rb:.2f}\n{stars}'
        ax4.text((x_left + x_right) / 2, bracket_h + tip_h * 0.5,
                 label_text, ha='center', va='bottom', fontsize=6)

    fig4.tight_layout()

    save_path4 = os.path.join(vis_dir, img_name + '_phase_delta_r2_pooled.pdf')
    fig4.savefig(save_path4, dpi=600, transparent=True, bbox_inches='tight', format='pdf')
    plt.close(fig4)

    return phase_r2


def plot_dsupr_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='lines',
    show_tick_labels=True):

    print('Plotting DSUP Ratio results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_dsupr_slow_all = []
    rSLDS_dsupr_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_dsupr = np.load(os.path.join(session_results_dir, rSLDS_dsupr_name + '.npz'))
        rSLDS_dsupr_slow = rSLDS_dsupr['dsupr_slow']
        rSLDS_dsupr_fast = rSLDS_dsupr['dsupr_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_dsupr_slow = rSLDS_dsupr_slow[..., -1]
        rSLDS_dsupr_fast = rSLDS_dsupr_fast[..., -1]

        rSLDS_dsupr_slow_all.append(rSLDS_dsupr_slow)
        rSLDS_dsupr_fast_all.append(rSLDS_dsupr_fast)

    ## Stack all sessions
    rSLDS_dsupr_slow_all = np.stack(rSLDS_dsupr_slow_all, axis=0)
    rSLDS_dsupr_fast_all = np.stack(rSLDS_dsupr_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_dsupr_slow_all_mean = np.mean(rSLDS_dsupr_slow_all, axis=(0, 2, 3))
    rSLDS_dsupr_fast_all_mean = np.mean(rSLDS_dsupr_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_dsupr_slow_all_se = np.std(rSLDS_dsupr_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_dsupr_fast_all_se = np.std(rSLDS_dsupr_fast_all, axis=(0, 2, 3)) / correction_factor

    dsupr_results = {
        trial_filters[1]:
            {
                'mean': rSLDS_dsupr_slow_all_mean,
                'se': rSLDS_dsupr_slow_all_se,
            },
        trial_filters[0]:
            {
                'mean': rSLDS_dsupr_fast_all_mean,
                'se': rSLDS_dsupr_fast_all_se,
            }
    }

    vmin = 0
    vmax = 0.6

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                dsupr_results[trial_filter]['mean'][train_or_test_id, ...], 
                ax=axs[i_tf], 
                cbar=True,
                vmin=vmin,
                vmax=vmax,
                cmap='viridis')

            axs[i_tf].set_title('DSUP Ratio')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].invert_yaxis()

        fig.tight_layout()

    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = dsupr_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('DSUP Ratio')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].set_zlim(vmin, vmax)

    elif form == 'lines':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):
            
            ## Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                dsupr_results[trial_filter]['mean'][train_or_test_id, ...],
                dsupr_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states, 
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)
            
            # axs[i_tf].axhline(
            #     y=performance_reference,
            #     color='black',
            #     lw=size_line_thin,
            #     alpha=alpha_line_thin,
            #     linestyle='--')

            for i_ds, n_discrete_state in reversed(list(enumerate(ns_discrete_states))):

                dsup_mean = dsupr_results[trial_filter]['mean'][train_or_test_id, :, i_ds]
                dsup_se = dsupr_results[trial_filter]['se'][train_or_test_id, :, i_ds]

                # axs[i_tf].plot(
                #     ns_states,
                #     dsupr_results[trial_filter]['mean'][train_or_test_id, :, i_ds],
                #     color=discrete_state_colors[i_ds],
                #     alpha=alpha_line,
                #     lw=0.5)
                
                axs[i_tf].fill_between(
                    ns_states,
                    dsup_mean - dsup_se, 
                    dsup_mean + dsup_se,
                    color=discrete_state_colors[i_ds], 
                    alpha=alpha_line, 
                    linewidth=0)
            
            xticks_range = [s for s in ns_states if 2 <= s <= 20]
            axs[i_tf].set_xticks(xticks_range)
            axs[i_tf].set_yticks([0.25, 0.50, 0.75])

            if show_tick_labels:
                tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(xticks_range)]
                axs[i_tf].set_xticklabels(tick_labels)
            else:
                axs[i_tf].set_xticklabels([])
                axs[i_tf].set_yticklabels([])

            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            # axs[i_tf].set_ylabel('DSUP Ratio', fontsize=5)
            axs[i_tf].set_ylim(0, 0.75)

            ## Remove top and right spines
            axs[i_tf].spines['top'].set_visible(False)
            axs[i_tf].spines['right'].set_visible(False)

        ## Add custom legend
        # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

        # custom_lines = [
        #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
        #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

        # legend_labels = ['1', '2', '3', '4', '6', '8', '12', '16']

        # ## Create legend with custom elements
        # plt.legend(
        #     custom_lines, 
        #     legend_labels, 
        #     title='# of Discrete States',
        #     loc='upper center', 
        #     bbox_to_anchor=(-0.2, -0.15), 
        #     fancybox=False, 
        #     shadow=False, 
        #     ncol=8)

    fig.tight_layout()

    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'dsupr',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=False, bbox_inches=None, format='pdf')
    plt.close(fig)


def plot_per_time_dsupr_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    truncate_percentile=10,
    visual_delay_time=0,
    show_tick_labels=True):

    print('Plotting dsupr results...')

    rSLDS_same_speed_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'same_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))
    
    rSLDS_cross_speed_dsupr_name = '_'.join(map(str, [x for x in [
        'dsupr_totf',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        'cross_speed',
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_same_speed_dsupr_slow_all = []
    rSLDS_same_speed_dsupr_fast_all = []
    rSLDS_cross_speed_dsupr_slow_all = []
    rSLDS_cross_speed_dsupr_fast_all = []
    trial_length_slow_min_all = []
    trial_length_fast_min_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_same_speed_dsupr = np.load(os.path.join(session_results_dir, rSLDS_same_speed_dsupr_name + '.npz'))
        rSLDS_same_speed_dsupr_slow = rSLDS_same_speed_dsupr['dsupr_slow_test_per_time']
        rSLDS_same_speed_dsupr_fast = rSLDS_same_speed_dsupr['dsupr_fast_test_per_time']
        trial_lengths_slow_same_speed = rSLDS_same_speed_dsupr['trial_lengths_slow'] - 1
        trial_lengths_fast_same_speed = rSLDS_same_speed_dsupr['trial_lengths_fast'] - 1

        rSLDS_cross_speed_dsupr = np.load(os.path.join(session_results_dir, rSLDS_cross_speed_dsupr_name + '.npz'))
        rSLDS_cross_speed_dsupr_slow = rSLDS_cross_speed_dsupr['dsupr_slow_test_per_time']
        rSLDS_cross_speed_dsupr_fast = rSLDS_cross_speed_dsupr['dsupr_fast_test_per_time']
        trial_lengths_slow_cross_speed = rSLDS_cross_speed_dsupr['trial_lengths_slow'] - 1
        trial_lengths_fast_cross_speed = rSLDS_cross_speed_dsupr['trial_lengths_fast'] - 1

        assert np.array_equal(trial_lengths_slow_same_speed, trial_lengths_slow_cross_speed)
        assert np.array_equal(trial_lengths_fast_same_speed, trial_lengths_fast_cross_speed)
        trial_lengths_slow = trial_lengths_slow_same_speed
        trial_lengths_fast = trial_lengths_fast_same_speed

        # Take the first discrete state (LDS) and last n_iters
        rSLDS_same_speed_dsupr_slow = rSLDS_same_speed_dsupr_slow[:, :, 0, -1]
        rSLDS_same_speed_dsupr_fast = rSLDS_same_speed_dsupr_fast[:, :, 0, -1]
        rSLDS_cross_speed_dsupr_slow = rSLDS_cross_speed_dsupr_slow[:, :, 0, -1]
        rSLDS_cross_speed_dsupr_fast = rSLDS_cross_speed_dsupr_fast[:, :, 0, -1]

        # Truncate the dsupr results with specified percentile
        rSLDS_same_speed_dsupr_slow, trial_length_slow_min = drop_and_truncate(rSLDS_same_speed_dsupr_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_same_speed_dsupr_fast, trial_length_fast_min = drop_and_truncate(rSLDS_same_speed_dsupr_fast, trial_lengths_fast, truncate_percentile)
        rSLDS_cross_speed_dsupr_slow, _ = drop_and_truncate(rSLDS_cross_speed_dsupr_slow, trial_lengths_slow, truncate_percentile)
        rSLDS_cross_speed_dsupr_fast, _ = drop_and_truncate(rSLDS_cross_speed_dsupr_fast, trial_lengths_fast, truncate_percentile)

        trial_length_slow_min_all.append(trial_length_slow_min)
        trial_length_fast_min_all.append(trial_length_fast_min)

        # Average over random states
        rSLDS_same_speed_dsupr_slow = np.mean(rSLDS_same_speed_dsupr_slow, axis=0)
        rSLDS_same_speed_dsupr_fast = np.mean(rSLDS_same_speed_dsupr_fast, axis=0)
        rSLDS_cross_speed_dsupr_slow = np.mean(rSLDS_cross_speed_dsupr_slow, axis=0)
        rSLDS_cross_speed_dsupr_fast = np.mean(rSLDS_cross_speed_dsupr_fast, axis=0)

        # Append to all sessions
        rSLDS_same_speed_dsupr_slow_all.append(rSLDS_same_speed_dsupr_slow)
        rSLDS_same_speed_dsupr_fast_all.append(rSLDS_same_speed_dsupr_fast)
        rSLDS_cross_speed_dsupr_slow_all.append(rSLDS_cross_speed_dsupr_slow)
        rSLDS_cross_speed_dsupr_fast_all.append(rSLDS_cross_speed_dsupr_fast)

    ## Combine all sessions

    # Get the minimum trial length across all sessions
    trial_length_slow_min = np.min(trial_length_slow_min_all)
    trial_length_fast_min = np.min(trial_length_fast_min_all)

    # Get the number of trials across all sessions
    n_trials_slow_all = np.sum([x.shape[1] for x in rSLDS_same_speed_dsupr_slow_all])
    n_trials_fast_all = np.sum([x.shape[1] for x in rSLDS_same_speed_dsupr_fast_all])

    n_ns_states = len(ns_states)
    rSLDS_same_speed_dsupr_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_same_speed_dsupr_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))
    rSLDS_cross_speed_dsupr_slow_all_ = np.zeros((n_ns_states, n_trials_slow_all, trial_length_slow_min))
    rSLDS_cross_speed_dsupr_fast_all_ = np.zeros((n_ns_states, n_trials_fast_all, trial_length_fast_min))

    n_trials_slow_cumulative = 0
    n_trials_fast_cumulative = 0
    for i in range(len(session_data_names)):
        n_trials_slow = rSLDS_same_speed_dsupr_slow_all[i].shape[1]
        n_trials_fast = rSLDS_same_speed_dsupr_fast_all[i].shape[1]

        # Truncate to the minimum trial length
        rSLDS_same_speed_dsupr_slow_all[i] = rSLDS_same_speed_dsupr_slow_all[i][..., :trial_length_slow_min]
        rSLDS_same_speed_dsupr_fast_all[i] = rSLDS_same_speed_dsupr_fast_all[i][..., :trial_length_fast_min]
        rSLDS_cross_speed_dsupr_slow_all[i] = rSLDS_cross_speed_dsupr_slow_all[i][..., :trial_length_slow_min]
        rSLDS_cross_speed_dsupr_fast_all[i] = rSLDS_cross_speed_dsupr_fast_all[i][..., :trial_length_fast_min]

        # Concatenate all sessions
        rSLDS_same_speed_dsupr_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_same_speed_dsupr_slow_all[i]
        rSLDS_same_speed_dsupr_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_same_speed_dsupr_fast_all[i]
        rSLDS_cross_speed_dsupr_slow_all_[:, n_trials_slow_cumulative:n_trials_slow_cumulative + n_trials_slow, :] = rSLDS_cross_speed_dsupr_slow_all[i]
        rSLDS_cross_speed_dsupr_fast_all_[:, n_trials_fast_cumulative:n_trials_fast_cumulative + n_trials_fast, :] = rSLDS_cross_speed_dsupr_fast_all[i]

        n_trials_slow_cumulative += n_trials_slow
        n_trials_fast_cumulative += n_trials_fast

    dsupr_results = {
        trial_filters[0]:
            {
                'LDS_mean_same_speed': rSLDS_same_speed_dsupr_fast_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_fast_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_dsupr_fast_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_fast_all[0],
            },
        trial_filters[1]:
            {
                'LDS_mean_same_speed': rSLDS_same_speed_dsupr_slow_all_,
                'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_slow_all_,
                # 'LDS_mean_same_speed': rSLDS_same_speed_dsupr_slow_all[0],
                # 'LDS_mean_cross_speed': rSLDS_cross_speed_dsupr_slow_all[0],
            }
    }
    
    vmin = 0
    vmax = 1
    
    ## Plot results

    fig1 = plt.figure(figsize=(24, 12))
    ax0 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig1.add_subplot(1, 2, 2, projection='3d')
    axs = [ax0, ax1]

    for i_tf, trial_filter in enumerate(trial_filters):

        # Plot lines and fill under them
        for i_cs, n_states in enumerate(ns_states):

            color_list = ['blue', 'red']

            zs_list = [
                np.mean(dsupr_results[trial_filter]['LDS_mean_same_speed'], axis=1)[i_cs],
                np.mean(dsupr_results[trial_filter]['LDS_mean_cross_speed'], axis=1)[i_cs]
            ]

            for color, zs in zip(color_list, zs_list):

                trial_length = zs.shape[-1]

                xs = np.arange(trial_length)
                ys = np.full(trial_length, n_states)

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=color, 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=color, 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

        if trial_filter in ['slow', 'far']:
            xtick_step = 5
        else:
            xtick_step = 1

        axs[i_tf].set_xticks(xs[::xtick_step])
        yticks_range = [s for s in ns_states if 2 <= s <= 20]
        axs[i_tf].set_yticks(yticks_range)
        axs[i_tf].set_zticks([0.25, 0.50, 0.75])

        if show_tick_labels:
            axs[i_tf].set_xticklabels(np.round(xs[::xtick_step] * config.time_step + visual_delay_time, 2))
        else:
            axs[i_tf].set_xticklabels([])
            axs[i_tf].set_yticklabels([])
            axs[i_tf].set_zticklabels([])


        # Set labels and title
        axs[i_tf].set_xlabel('Time (s)', fontsize=label_fontsize)
        axs[i_tf].set_ylabel('# of Latent States', fontsize=label_fontsize)
        axs[i_tf].set_zlabel('Absolute Angle Error (degrees)', fontsize=label_fontsize)

        # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
        # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
        axs[i_tf].set_zlim(vmin, vmax)

        # axs[i_tf].set_yticklabels(['PCA'] + ns_discrete_states)

        # Adjust viewing angle
        # axs[i_tf].view_init(elev=10, azim=-75)
        axs[i_tf].view_init(elev=20, azim=-40)

        # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

        ## Remove background panes
        axs[i_tf].xaxis.pane.fill = False
        axs[i_tf].yaxis.pane.fill = False
        axs[i_tf].zaxis.pane.fill = False

        ## Set grid colors
        # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
        # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

        # ## Create tick labels: label only at even indices, else empty string
        # tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

        # ## Set tick labels
        # axs[i_tf].set_xticklabels(tick_labels)


    ## Add custom legend
    # fig.subplots_adjust(bottom=0.2)  ## Increase the bottom margin

    # custom_lines = [
    #     Line2D([0], [0], color='blue',                   lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[0], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[1], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[2], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[3], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[4], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[5], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[6], lw=2, linestyle='-'),
    #     Line2D([0], [0], color=discrete_state_colors[7], lw=2, linestyle='-')]

    # legend_labels = ['PCA', '1', '2', '3', '4', '6', '8', '12', '16']

    # ## Create legend with custom elements
    # plt.legend(
    #     custom_lines, 
    #     legend_labels, 
    #     title='# of Discrete States',
    #     loc='upper center', 
    #     bbox_to_anchor=(-0.2, -0.15), 
    #     fancybox=False, 
    #     shadow=False, 
    #     ncol=9)

    ## Write image
    if len(session_data_names) > 1:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'per_time_dsupr',
        truncate_percentile,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        dynamics_class,
        emission_class,
        init_type,
        alpha] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig1.savefig(save_path, dpi=600, transparent=True)
    # plt.show()
    # print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    # print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig1)


def plot_elbos_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    n_iters,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap'):

    print('Plotting ELBOs...')

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_elbos_name = '_'.join(map(str, [x for x in [
        'elbos',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        # trial_filters,
        ['slow', 'fast'],
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        n_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_elbos_slow_all = []
    rSLDS_elbos_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_elbos = np.load(os.path.join(session_results_dir, rSLDS_elbos_name + '.npz'))
        rSLDS_elbos_slow = rSLDS_elbos['elbos_slow']
        rSLDS_elbos_fast = rSLDS_elbos['elbos_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_elbos_slow = rSLDS_elbos_slow[..., -1]
        rSLDS_elbos_fast = rSLDS_elbos_fast[..., -1]

        rSLDS_elbos_slow_all.append(rSLDS_elbos_slow)
        rSLDS_elbos_fast_all.append(rSLDS_elbos_fast)

    ## Stack all sessions
    rSLDS_elbos_slow_all = np.stack(rSLDS_elbos_slow_all,  axis=0)
    rSLDS_elbos_fast_all = np.stack(rSLDS_elbos_fast_all,  axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_elbos_slow_all_mean = np.mean(rSLDS_elbos_slow_all, axis=(0, 2, 3))
    rSLDS_elbos_fast_all_mean = np.mean(rSLDS_elbos_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_elbos_slow_all_se = np.std(rSLDS_elbos_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_elbos_fast_all_se = np.std(rSLDS_elbos_fast_all, axis=(0, 2, 3)) / correction_factor


    elbos_results = {
        'slow':
            {
                'mean': rSLDS_elbos_slow_all_mean,
                'se': rSLDS_elbos_slow_all_se,
            },
        'fast':
            {
                'mean': rSLDS_elbos_fast_all_mean,
                'se': rSLDS_elbos_fast_all_se,
            }
    }

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    vmin_all = np.min([np.min(elbos_results['slow']['mean'][train_or_test_id]), np.min(elbos_results['fast']['mean'][train_or_test_id])])
    vmax_all = np.max([np.max(elbos_results['slow']['mean'][train_or_test_id]), np.max(elbos_results['fast']['mean'][train_or_test_id])])

    vmin = vmin_all - 10
    vmax = vmax_all + 10


    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                elbos_results[trial_filter]['mean'][train_or_test_id, ...], 
                ax=axs[i_tf], 
                cbar=True,
                # vmin=vmin,
                # vmax=vmax,
                cmap='viridis')

            axs[i_tf].set_title('ELBOs')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            axs[i_tf].set_xticklabels(ns_discrete_states)
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].invert_yaxis()

        fig.tight_layout()


    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = elbos_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('ELBOs')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            # axs[i_tf].set_zlim(vmin, vmax)


    elif form == 'waterfall':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            # Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                elbos_results[trial_filter]['mean'][train_or_test_id, ...],
                elbos_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states,
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)

            # Plot lines and fill under them
            for i_cs in range(len(ns_states)):

                xs = np.arange(len(ns_discrete_states))
                ys = np.full_like(ns_discrete_states, i_cs)
                zs = elbos_results[trial_filter]['mean'][train_or_test_id, i_cs]

                # Plot the line
                axs[i_tf].plot(
                    xs, 
                    ys, 
                    zs, 
                    color=continuous_state_colors[i_cs], 
                    lw=size_line,
                    alpha=alpha_line)

                # Create the polygon for filling
                verts = utils_vis.polygon_under_curve(xs, ys, zs, z_min=vmin)

                # Create a Poly3DCollection object
                poly = Poly3DCollection(
                    verts, 
                    facecolor=continuous_state_colors[i_cs], 
                    alpha=alpha_line_thin)

                # Add the polygon to the axes
                axs[i_tf].add_collection3d(poly)

            # Set labels and title
            axs[i_tf].set_ylabel('# of Continuous States', fontsize=label_fontsize)
            axs[i_tf].set_xlabel('# of Discrete States', fontsize=label_fontsize)
            axs[i_tf].set_zlabel('ELBOs', fontsize=label_fontsize)

            # axs[i_tf].set_xlim(ns_states[0], ns_states[-1])
            # axs[i_tf].set_ylim(ns_discrete_states[0], ns_discrete_states[-1])
            axs[i_tf].set_zlim(vmin, vmax)

            axs[i_tf].set_yticks(np.arange(len(ns_states)))
            axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xticks(np.arange(len(ns_discrete_states)))
            axs[i_tf].set_xticklabels(ns_discrete_states)

            # Adjust viewing angle
            axs[i_tf].view_init(elev=20, azim=-40)

            # axs[i_tf].tick_params(axis='both', which='major', labelsize=label_fontsize)

            ## Remove background panes
            axs[i_tf].xaxis.pane.fill = False
            axs[i_tf].yaxis.pane.fill = False
            axs[i_tf].zaxis.pane.fill = False

            ## Set grid colors
            # axs[i_tf].xaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].yaxis._axinfo['grid'].update(color='black')
            # axs[i_tf].zaxis._axinfo['grid'].update(color='black')    

            # ## Create tick labels: label only at even indices, else empty string
            tick_labels = [str(x) if i % 2 == 0 else '' for i, x in enumerate(ns_states)]

            # ## Set tick labels
            axs[i_tf].set_yticklabels(tick_labels)

            ## Visualize the reference plane
            # Create 2D meshgrid for x (discrete states) and y (continuous states)
            X, Y = np.meshgrid(
                np.arange(len(ns_discrete_states)),  # x-coordinates
                np.arange(len(ns_states))            # y-coordinates
            )

            # Z will be filled with the reference-performance value
            Z = np.full_like(X, performance_reference, dtype=float)
            # ipdb.set_trace()
            print(performance_reference)

            axs[i_tf].plot_surface(
                X,      # x-coordinates
                Y,      # y-coordinates
                Z,      # z-coordinates
                color='gray',
                alpha=0.3,
                shade=False
            )

        
    ## Write image
    if len(session_data_names) > 3:
        session_data_names_str = str(len(session_data_names)) + '_sessions'
    else:
        session_data_names_str = str(session_data_names)

    img_name = '_'.join(map(str, [x for x in [
        session_data_names_str,
        'elbos',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=False)
    # plt.show()
    print('(Axis 0) Elevation angle:', axs[0].elev, ' Azimuth angle:', axs[0].azim)
    print('(Axis 1) Elevation angle:', axs[1].elev, ' Azimuth angle:', axs[1].azim)
    plt.close(fig)


def plot_entropy_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='heatmap'):

    print('Plotting Entropy Difference results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    rSLDS_entropy_name = '_'.join(map(str, [x for x in [
        'entropy',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_entropy_slow_all = []
    rSLDS_entropy_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        rSLDS_entropy = np.load(os.path.join(session_results_dir, rSLDS_entropy_name + '.npz'))

        rSLDS_entropy_slow = rSLDS_entropy['entropy_per_trial_slow'] - rSLDS_entropy['entropy_per_time_slow']
        rSLDS_entropy_fast = rSLDS_entropy['entropy_per_trial_fast'] - rSLDS_entropy['entropy_per_time_fast']

        ## Take the last n_iters, then add to all sessions
        rSLDS_entropy_slow = rSLDS_entropy_slow[..., -1]
        rSLDS_entropy_fast = rSLDS_entropy_fast[..., -1]

        rSLDS_entropy_slow_all.append(rSLDS_entropy_slow)
        rSLDS_entropy_fast_all.append(rSLDS_entropy_fast)

    ## Stack all sessions
    rSLDS_entropy_slow_all = np.stack(rSLDS_entropy_slow_all, axis=0)
    rSLDS_entropy_fast_all = np.stack(rSLDS_entropy_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    rSLDS_entropy_slow_all_mean = np.mean(rSLDS_entropy_slow_all, axis=(0, 2, 3))
    rSLDS_entropy_fast_all_mean = np.mean(rSLDS_entropy_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_entropy_slow_all_se = np.std(rSLDS_entropy_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_entropy_fast_all_se = np.std(rSLDS_entropy_fast_all, axis=(0, 2, 3)) / correction_factor

    entropy_results = {
        trial_filters[1]:
            {
                'mean': rSLDS_entropy_slow_all_mean,
                'se': rSLDS_entropy_slow_all_se,
            },
        trial_filters[0]:
            {
                'mean': rSLDS_entropy_fast_all_mean,
                'se': rSLDS_entropy_fast_all_se,
            }
    }

    vmin = -0.5
    vmax = 0.5

    if train_or_test == 'train':
        train_or_test_id = 0
    elif train_or_test == 'test':
        train_or_test_id = 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm))

        for i_tf, trial_filter in enumerate(trial_filters):

            # Find the simplest model with performance better than best result + its standard error
            performance_reference = utils_vis.model_selection(
                entropy_results[trial_filter]['mean'][train_or_test_id, ...],
                entropy_results[trial_filter]['se'][train_or_test_id, ...],
                ns_states,
                ns_discrete_states,
                includes_pca=False,
                higher_is_better=True)
            print(performance_reference)
            
            sns.heatmap(
                entropy_results[trial_filter]['mean'][train_or_test_id, :, 1:].T, # Exclude LDS
                ax=axs[i_tf],
                vmin=vmin,
                vmax=vmax,
                cmap='mako',
                cbar=False,          # ← suppress the color-bar
                # square=True,         # ← make the axes a true square
                xticklabels=False,   # ← drop x-axis tick labels
                yticklabels=False,   # ← drop y-axis tick labels
                linewidths=0,
                linecolor=None,
                rasterized=True,
            )
        
            # If you also want to hide the tick marks themselves:
            axs[i_tf].tick_params(left=False, bottom=False)
            axs[i_tf].set_box_aspect(1)

            # axs[i_tf].set_title('Entropy Difference')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_discrete_states[1:])
            # axs[i_tf].set_xticklabels(ns_states)
            # axs[i_tf].set_ylabel('Number of discrete states', fontsize=5)
            # axs[i_tf].set_xlabel('Number of continuous states', fontsize=5)
            axs[i_tf].invert_yaxis()

            # axs[i_tf].set_aspect('equal')

        fig.tight_layout()


    elif form == 'surface':

        fig = plt.figure(figsize=(24, 12))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax0, ax1]

        for i_tf, trial_filter in enumerate(trial_filters):

            data = entropy_results[trial_filter]['mean'][train_or_test_id, ...]
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            axs[i_tf].plot_surface(
                x, y, data,
                cmap='viridis',
                alpha=0.7
            )

            axs[i_tf].set_title('Entropy Difference')
            # axs[i_tf].set_xticklabels(['LDS'] + ns_discrete_states)
            # axs[i_tf].set_yticklabels(ns_states)
            axs[i_tf].set_xlabel('# of Discrete States')
            axs[i_tf].set_ylabel('# of Continuous States')
            axs[i_tf].set_zlim(vmin, vmax)


    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'entropy',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    plt.close(fig)


    # Create a continuous colormap from seaborn's 'rocket' palette
    cmap = sns.color_palette("mako", as_cmap=True)
    norm = Normalize(vmin=2, vmax=20)

    # Build the figure
    fig, ax = plt.subplots(figsize=(20*mm, 2*mm))
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add a horizontal colorbar without ticks or labels
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    # Remove the axes frame completely
    ax.set_frame_on(False)
    fig.savefig(os.path.join(vis_dir, 'entropy_color_bar.pdf'), bbox_inches=None, transparent=True, dpi=600, format='pdf')


def plot_skew_ratio_results_avg_session(
    unit_filter,
    input_unit_filter,
    data_format,
    train_test_option,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha,
    train_or_test='test',
    form='lines'):

    print('Plotting Skew-Symmetric Ratio results...')

    task_name = session_data_names[0].split('_')[-1]

    xticks = np.arange(len(ns_states))
    xticklabels = ns_states

    ## Load rSLDS results
    rSLDS_skew_name = '_'.join(map(str, [x for x in [
        'skew_ratio',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'rSLDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    ## Load LDS results
    LDS_skew_name = '_'.join(map(str, [x for x in [
        'skew_ratio',
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        random_states,
        n_folds,
        ns_states,
        ns_discrete_states,
        ns_iters,
        'LDS',
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha] if x is not None]))

    rSLDS_skew_slow_all = []
    rSLDS_skew_fast_all = []
    LDS_skew_slow_all = []
    LDS_skew_fast_all = []

    for session_data_name in session_data_names:

        session_results_dir = os.path.join(results_dir, session_data_name)

        ## rSLDS
        rSLDS_skew = np.load(os.path.join(session_results_dir, rSLDS_skew_name + '.npz'))
        rSLDS_skew_slow = rSLDS_skew['skew_ratio_slow']
        rSLDS_skew_fast = rSLDS_skew['skew_ratio_fast']

        ## Take the last n_iters
        rSLDS_skew_slow = rSLDS_skew_slow[..., -1]
        rSLDS_skew_fast = rSLDS_skew_fast[..., -1]

        rSLDS_skew_slow_all.append(rSLDS_skew_slow)
        rSLDS_skew_fast_all.append(rSLDS_skew_fast)

        ## LDS
        try:
            LDS_skew = np.load(os.path.join(session_results_dir, LDS_skew_name + '.npz'))
            LDS_skew_slow = LDS_skew['skew_ratio_slow']
            LDS_skew_fast = LDS_skew['skew_ratio_fast']

            LDS_skew_slow = LDS_skew_slow[..., -1]
            LDS_skew_fast = LDS_skew_fast[..., -1]

            LDS_skew_slow_all.append(LDS_skew_slow)
            LDS_skew_fast_all.append(LDS_skew_fast)
        except FileNotFoundError:
            pass

    ## Stack all sessions
    rSLDS_skew_slow_all = np.stack(rSLDS_skew_slow_all, axis=0)
    rSLDS_skew_fast_all = np.stack(rSLDS_skew_fast_all, axis=0)

    ## Compute mean and standard error over sessions, random states, and folds
    ## Dimensions after stacking: (sessions, train/test, random_states, folds, n_cont_states, n_disc_states)
    rSLDS_skew_slow_all_mean = np.nanmean(rSLDS_skew_slow_all, axis=(0, 2, 3))
    rSLDS_skew_fast_all_mean = np.nanmean(rSLDS_skew_fast_all, axis=(0, 2, 3))

    correction_factor = np.sqrt(len(session_data_names) * len(random_states) * n_folds)
    rSLDS_skew_slow_all_se = np.nanstd(rSLDS_skew_slow_all, axis=(0, 2, 3)) / correction_factor
    rSLDS_skew_fast_all_se = np.nanstd(rSLDS_skew_fast_all, axis=(0, 2, 3)) / correction_factor

    skew_results = {
        trial_filters[1]: {
            'mean': rSLDS_skew_slow_all_mean,
            'se': rSLDS_skew_slow_all_se,
        },
        trial_filters[0]: {
            'mean': rSLDS_skew_fast_all_mean,
            'se': rSLDS_skew_fast_all_se,
        }
    }

    train_or_test_id = 0 if train_or_test == 'train' else 1

    ## Plot results
    if form == 'heatmap':

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm))

        for i_tf, trial_filter in enumerate(trial_filters):

            sns.heatmap(
                skew_results[trial_filter]['mean'][train_or_test_id, ...].T,
                ax=axs[i_tf],
                cmap='mako',
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
                linecolor=None,
                rasterized=True,
            )

            axs[i_tf].tick_params(left=False, bottom=False)
            axs[i_tf].set_box_aspect(1)
            axs[i_tf].invert_yaxis()

        fig.tight_layout()

    else:  # form == 'lines'

        fig, axs = plt.subplots(1, 2, figsize=(45*mm, 22.5*mm), sharey=True)

        for i_tf, trial_filter in enumerate(trial_filters):

            for i_ds in reversed(range(len(ns_discrete_states))):

                skew_mean = skew_results[trial_filter]['mean'][train_or_test_id, :, i_ds]
                skew_se = skew_results[trial_filter]['se'][train_or_test_id, :, i_ds]

                axs[i_tf].fill_between(
                    ns_states,
                    skew_mean - skew_se,
                    skew_mean + skew_se,
                    color=discrete_state_colors[i_ds],
                    alpha=alpha_line,
                    linewidth=0)

            axs[i_tf].set_xticks(ns_states)
            axs[i_tf].set_xticklabels([])

            axs[i_tf].spines['top'].set_visible(False)
            axs[i_tf].spines['right'].set_visible(False)

        fig.tight_layout()

    ## Write image
    session_data_names_str = str(len(session_data_names)) + '_sessions'

    img_name = '_'.join(map(str, [x for x in [
        task_name,
        session_data_names_str,
        'skew_ratio',
        train_or_test,
        unit_filter,
        input_unit_filter,
        window_config,
        time_offset,
        data_format,
        trial_filters,
        train_test_option,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha,
        form] if x is not None]))

    save_path = os.path.join(vis_dir, img_name + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, bbox_inches=None, format='pdf')
    plt.close(fig)


if __name__ == '__main__':
    
    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     label_format,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         label_formats,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
    #     plot_per_time_decoding_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         label_format,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         truncate_percentile=10,
    #         visual_delay_time=0.1)
    
    for (
        unit_filter,
        input_unit_filter,
        data_format,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha) in itertools.product(
            unit_filters,
            input_unit_filters,
            data_formats,
            dynamics_classes,
            emission_classes,
            init_types,
            subspace_types,
            alphas):
        
        # LDS (K=1)
        # plot_per_time_inference_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     inference_type='forecast',
        #     truncate_percentile=10,
        #     visual_delay_time=0.132,
        #     peak_time=0.186,
        #     discrete_state_idx=0)

        # rSLDS (K=2)
        # plot_per_time_inference_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     inference_type='forecast',
        #     truncate_percentile=10,
        #     visual_delay_time=0.132,
        #     discrete_state_idx=1)

        # rSLDS phase analysis (K=2): transient vs steady
        # plot_rslds_phase_inference_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     n_discrete=2,
        #     inference_type='forecast',
        #     truncate_percentile=10,
        #     visual_delay_time=0.132)

        # LDS phase analysis: transient vs steady by peak_time
        plot_lds_phase_inference_results_avg_session(
            unit_filter,
            input_unit_filter,
            data_format,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha,
            inference_type='forecast',
            truncate_percentile=10,
            visual_delay_time=0.132,
            peak_time=0.186)

        # plot_per_time_dsupr_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     truncate_percentile=10,
        #     visual_delay_time=0.25)
    
    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     label_format,
    #     train_test_option,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         label_formats,
    #         train_test_options,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):
        
    #     plot_decoding_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         label_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         form='scatter',
    #         show_tick_labels=False)

    # for (
    #     unit_filter,
    #     input_unit_filter,
    #     data_format,
    #     train_test_option,
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha) in itertools.product(
    #         unit_filters,
    #         input_unit_filters,
    #         data_formats,
    #         train_test_options,
    #         dynamics_classes,
    #         emission_classes,
    #         init_types,
    #         subspace_types,
    #         alphas):

    #     plot_inference_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         inference_type='forecast',
    #         form='heatmap')
        
    #     plot_dsupr_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         form='lines',
    #         show_tick_labels=False,
    #         )
        
    # #     # plot_elbos_avg_session(
    # #     #     unit_filter,
    # #     #     input_unit_filter,
    # #     #     data_format,
    # #     #     train_test_option,
    # #     #     25,
    # #     #     dynamics_class,
    # #     #     emission_class,
    # #     #     init_type,
    # #     #     subspace_type,
    # #     #     alpha,
    # #     #     train_or_test='test',
    # #     #     form='waterfall')

        # plot_entropy_results_avg_session(
        #     unit_filter,
        #     input_unit_filter,
        #     data_format,
        #     train_test_option,
        #     dynamics_class,
        #     emission_class,
        #     init_type,
        #     subspace_type,
        #     alpha,
        #     train_or_test='test',
        #     form='heatmap')

    #     plot_skew_ratio_results_avg_session(
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         train_test_option,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha,
    #         train_or_test='test',
    #         form='heatmap')
