import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.collections as mcoll
from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon, mannwhitneyu, sem, t
from prettytable import PrettyTable

import dynamical_systems_analyses.utils.utils_vis as utils_vis
import ipdb

from vis_config import *



vis_dir = '/Users/ynffsy/Documents/andersen_lab_local/visualizations/dynamics_paper'
# vis_dir = '/home/ynffsy/Desktop/andersen_lab/visualizations/dynamics_paper'

N1_sessions = [
    'sub-N1_ses-20190412_tf_CenterStart',
    'sub-N1_ses-20190517_tf_CenterStart',
    'sub-N1_ses-20190528_tf_CenterStart',
]

N2_sessions = [
    'sub-N2_ses-20240516_tf_CenterStart',
    'sub-N2_ses-20240530_tf_CenterStart',
    'sub-N2_ses-20240816_tf_CenterStart',
    'sub-N2_ses-20240820_tf_CenterStart',
    'sub-N2_ses-20241015_tf_CenterStart',
    'sub-N2_ses-20241022_tf_CenterStart',
]

N2_RadialGrid_sessions = [
    'sub-N2_ses-20241105_tf_RadialGrid',
    'sub-N2_ses-20241211_tf_RadialGrid',
    'sub-N2_ses-20250408_tf_RadialGrid',
]

N2_CenterStartInterleave_sessions = [
    'sub-N2_ses-20250417_tf_CenterStartInterleave',
    'sub-N2_ses-20250422_tf_CenterStartInterleave',
    'sub-N2_ses-20250509_tf_CenterStartInterleave',
]

def construct_df(
    sessions, 
    trial_filters,
    unit_filters,
    window_config,
    data_format):
    
    df_list = []

    for session in sessions:
        # Construct the pkl file path
        session_vis_dir = os.path.join(vis_dir, session)

        for unit_filter in unit_filters:
            
            ## Old format
            # save_name = '_'.join(map(str, [x for x in [
            #     'crossnobis_RDM_superdiagonal',
            #     unit_filter,
            #     window_config,
            #     data_format,
            #     trial_filters] if x is not None]))

            # save_name += '_superdiagonal1'

            ## New format
            save_name = '_'.join(map(str, [x for x in [
                session,
                'crossnobis_RDM_superdiagonal_split',
                unit_filter,
                window_config] if x is not None]))

            save_name += '_superdiagonal1'
            save_name += '_nstd1.5'

            pkl_file = os.path.join(session_vis_dir, save_name + '_stats.pkl')

            # print(f"Loading {pkl_file}")

            # Load the dictionary: res_session = {'slow': slow_res, 'fast': fast_res}
            with open(pkl_file, 'rb') as f:
                res_session = pkl.load(f)

            # For each condition ('slow' and 'fast'), build a DataFrame and append
            for trial_filter in trial_filters:
                # res_session[condition] is a dict with keys: 
                # 'peak_onset_time', 'peak_time', 'peak_duration', 
                # 'left_base_time', 'peak_value'.
                temp_df = pd.DataFrame(res_session[trial_filter])

                # Add columns to identify the session and the condition (if desired)
                temp_df['session'] = session
                temp_df['unit_filter'] = unit_filter
                temp_df['trial_filter'] = trial_filter

                # Collect in a list for later concatenation
                df_list.append(temp_df)

    # Combine all dataframes into one
    df_sessions = pd.concat(df_list, ignore_index=True)

    return df_sessions


# def label_group(row, experiment):
#     """
#     Given a row (from one of the dataframes) and the experiment name
#     (e.g., 'N1', 'N2', or 'N2_Radial'), return the string label
#     you want on the boxplot x-axis.
#     """
#     trial_filter = row['trial_filter']
#     unit_filter  = row.get('unit_filter', 'MC')  # default MC for N1

#     if experiment == 'N1':
#         # N1 ballistic vs. sustained
#         if trial_filter == 'fast':
#             return 'N1 ballistic'
#         else:  # 'slow'
#             return 'N1 sustained'

#     elif experiment == 'N2':
#         # N2 can be ballistic vs. sustained, possibly also split by MC-LAT vs. MC-MED,
#         # and also a combined label ignoring unit_filter if you want that too.
#         # But you mentioned 6 distinct labels for N2:
#         #    - N2 MCL CO ballistic
#         #    - N2 MCL CO sustained
#         #    - N2 MCM CO ballistic
#         #    - N2 MCM CO sustained
#         #    - N2 ballistic        (unit_filter combined)
#         #    - N2 sustained        (unit_filter combined)
#         # We'll decide them as follows:

#         # For the MC-LAT / MC-MED labels:
#         if unit_filter == 'MC-LAT' and trial_filter == 'fast':
#             return 'N2 MCL CO ballistic'
#         elif unit_filter == 'MC-LAT' and trial_filter == 'slow':
#             return 'N2 MCL CO sustained'
#         elif unit_filter == 'MC-MED' and trial_filter == 'fast':
#             return 'N2 MCM CO ballistic'
#         elif unit_filter == 'MC-MED' and trial_filter == 'slow':
#             return 'N2 MCM CO sustained'
#         else:
#             # If, for some reason, you need the combined categories ignoring unit_filter,
#             # you could do that as well, but typically you'd put that in a separate row or
#             # a separate df. In your bullet points, you listed "N2 ballistic" and "N2 sustained"
#             # as well. If you truly want separate boxes for those too, you'd have to *duplicate*
#             # rows. But let's assume we only want the more specific 4 categories plus the combined 2.

#             # Example: we also generate "N2 ballistic" or "N2 sustained" labels by ignoring unit_filter:
#             if trial_filter == 'fast':
#                 return 'N2 ballistic'
#             else:
#                 return 'N2 sustained'

#     elif experiment == 'N2_RadialGrid':
#         # N2 RadialGrid has "near" = ballistic, "far" = sustained
#         # and we also have MC-LAT, MC-MED, plus combined categories.
#         if unit_filter == 'MC-LAT' and trial_filter == 'near':
#             return 'N2 MCL RG near'
#         elif unit_filter == 'MC-LAT' and trial_filter == 'far':
#             return 'N2 MCL RG far'
#         elif unit_filter == 'MC-MED' and trial_filter == 'near':
#             return 'N2 MCM RG near'
#         elif unit_filter == 'MC-MED' and trial_filter == 'far':
#             return 'N2 MCM RG far'
#         else:
#             # Combined ignoring unit_filter
#             if trial_filter == 'near':
#                 return 'N2 RadialGrid near'
#             else:
#                 return 'N2 RadialGrid far'

#     # Fallback if something unexpected
#     return 'Unknown'


def label_group(row, experiment):
    """
    Given a row (from one of the dataframes) and the experiment name
    (e.g., 'N1', 'N2', or 'N2_Radial'), return the string label
    you want on the boxplot x-axis.
    """
    unit_filter  = row.get('unit_filter', 'MC')  # default MC for N1

    if experiment == 'N1':
        return 'N1 MC CO'

    elif experiment == 'N2':
        if unit_filter == 'MC-LAT':
            return 'N2 MCL CO'
        elif unit_filter == 'MC-MED':
            return 'N2 MCM CO'
        else:
            return 'Unknown'

    elif experiment == 'N2_RadialGrid':
        if unit_filter == 'MC-LAT':
            return 'N2 MCL RG'
        elif unit_filter == 'MC-MED':
            return 'N2 MCM RG'
        else:
            return 'Unknown'
    
    elif experiment == 'N2_CenterStartInterleave':
        if unit_filter == 'MC-LAT':
            return 'N2 MCL COI'
        elif unit_filter == 'MC-MED':
            return 'N2 MCM COI'
        else:
            return 'Unknown'

    return 'Unknown'


def paired_95_CI(array1, array2):

    ## Compute 95% CI of mean difference for paired data
    differences = array1 - array2
    mean_diff = np.mean(differences)
    sem_diff = sem(differences)  # = np.std(differences, ddof=1) / np.sqrt(len(differences))

    # 4. Compute the t-critical value for a 95% CI with (n-1) degrees of freedom
    df = len(differences) - 1
    t_crit = t.ppf(1 - 0.025, df)  # two-tailed => alpha/2

    # 5. Calculate the margin of error and the confidence interval
    margin_of_error = t_crit * sem_diff
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"95% CI of Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")


def unpaired_95_CI(array1, array2):

    ## Compute 95% CI of mean difference for unpaired data
    # 1. Compute sample means
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    mean_diff = mean1 - mean2

    # 2. Compute sample variances
    var1 = np.var(array1, ddof=1)
    var2 = np.var(array2, ddof=1)

    n1 = len(array1)
    n2 = len(array2)

    # 3. Standard error (Welch's formula)
    se_diff = np.sqrt(var1/n1 + var2/n2)

    # 4. Degrees of freedom (Welch–Satterthwaite equation)
    df_numer = (var1/n1 + var2/n2)**2
    df_denom = (var1/n1)**2 / (n1 - 1) + (var2/n2)**2 / (n2 - 1)
    df = df_numer / df_denom  # effective degrees of freedom

    # 5. Critical t-value for a 95% CI
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df)

    # 6. Margin of error
    margin = t_crit * se_diff

    # 7. Confidence interval
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    print(f"Difference of means: {mean_diff:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


def plot_stats(
    df, 
    feature, 
    y_max=None,
    pairs_to_test=None,
    annotate_stats=False,
    y_label="New Y-Axis Label",
    formersci_notation=False,
    suffix='default',
    show_x_labels=True):

    theme_coral_light  = '#FFCCCC'
    theme_coral_mid    = '#F79C9C'
    theme_coral_dark   = '#E86B6B'
    theme_green_light  = '#CEF2EE'
    theme_green_mid    = '#78CCC2'
    theme_green_dark   = '#4ABDAF'
    theme_blue_light   = '#CBE7F5'
    theme_blue_mid     = '#87CCE6'
    theme_blue_dark    = '#52ABCC'
    theme_orange_light = '#FFE3C8'
    theme_orange_mid   = '#FDA058'
    theme_orange_dark  = '#E97123'
    theme_yellow_light = '#FFF4BD'
    theme_yellow_mid   = '#F7E68D'
    theme_yellow_dark  = '#F0D75B'
    theme_indigo_light = '#CCDBFD'
    theme_indigo_mid   = '#89A8F4'
    theme_indigo_dark  = '#5D86E6'

    theme_coral  = [theme_coral_light,  theme_coral_mid,  theme_coral_dark]
    theme_green  = [theme_green_light,  theme_green_mid,  theme_green_dark]
    theme_blue   = [theme_blue_light,   theme_blue_mid,   theme_blue_dark]
    theme_orange = [theme_orange_light, theme_orange_mid, theme_orange_dark]
    theme_yellow = [theme_yellow_light, theme_yellow_mid, theme_yellow_dark]
    theme_indigo = [theme_indigo_light, theme_indigo_mid, theme_indigo_dark]
    
    df.trial_filter = df.trial_filter.replace('fast', 'ballistic')
    df.trial_filter = df.trial_filter.replace('slow', 'sustained')

    df_CenterStart = df[df.task == 'CenterStart']
    df_RadialGrid = df[df.task == 'RadialGrid']
    df_CenterStartInterleave = df[df.task == 'CenterStartInterleave']

    if show_x_labels:
        # img_height=45*mm
        img_height=40*mm
    else:
        # img_height=38*mm
        img_height=33*mm

    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3,
        sharey=True,
        figsize=(90*mm, img_height),
        gridspec_kw={"width_ratios": [3, 2, 2]}
    )
    
    sns.violinplot(
        ax=ax1,
        data=df_CenterStart, 
        x='group', 
        y=feature,
        cut=0, # If you don't want the violin "tails" to extend beyond min/max data
        inner='quartile',
        palette={'ballistic':theme_blue[1], 'sustained':theme_orange[1]},
        split=True,
        hue='trial_filter',
        linewidth=0.25,
        saturation=1,
        alpha=0.8,
    )

    # Remove the violinplot edges
    for art in ax1.findobj(mcoll.PolyCollection):
        art.set_edgecolor("none")
    # Set the median line to be solid
    for l in ax1.lines[1::3]:
        l.set_linestyle('-')
    # Set other lines to be dashed
    for l in ax1.lines[::3]:
        l.set_linestyle('--')
    for l in ax1.lines[2::3]:
        l.set_linestyle('--')

    sns.violinplot(
        ax=ax2,
        data=df_CenterStartInterleave, 
        x='group', 
        y=feature,
        cut=0, # If you don't want the violin "tails" to extend beyond min/max data
        inner='quartile',
        palette={'ballistic':theme_indigo[1], 'sustained':theme_yellow[1]},
        split=True,
        hue='trial_filter',
        linewidth=0.25,
        saturation=1,
        alpha=0.8,
    )

    # Remove the violinplot edges
    for art in ax2.findobj(mcoll.PolyCollection):
        art.set_edgecolor("none")
    # Set the median line to be solid
    for l in ax2.lines[1::3]:
        l.set_linestyle('-')
    # Set other lines to be dashed
    for l in ax2.lines[::3]:
        l.set_linestyle('--')
    for l in ax2.lines[2::3]:
        l.set_linestyle('--')

    sns.violinplot(
        ax=ax3,
        data=df_RadialGrid, 
        x='group', 
        y=feature,
        cut=0, # If you don't want the violin "tails" to extend beyond min/max data
        inner='quartile',
        palette={'near':theme_green[1], 'far':theme_coral[1]},
        split=True,
        hue='trial_filter',
        linewidth=0.25,
        saturation=1,
        alpha=0.8,
    )

    # Remove the violinplot edges
    for art in ax3.findobj(mcoll.PolyCollection):
        art.set_edgecolor("none")
    # Set the median line to be solid
    for l in ax3.lines[1::3]:
        l.set_linestyle('-')
    # Set other lines to be dashed
    for l in ax3.lines[::3]:
        l.set_linestyle('--')
    for l in ax3.lines[2::3]:
        l.set_linestyle('--')

    # Rotate x-axis labels
    if show_x_labels:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=5)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=5)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=5)
    else:
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])

    # Remove titles
    ax1.set_title("")
    ax2.set_title("")
    ax3.set_title("")

    # Remove the top and right spines
    sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
    sns.despine(ax=ax2, top=True, right=True, left=True,  bottom=False)
    sns.despine(ax=ax3, top=True, right=True, left=True,  bottom=False)

    # Remove the legend titles
    # legend = ax1.legend()
    # legend.set_title(None)
    # legend = ax2.legend()
    # legend.set_title(None)
    # legend = ax3.legend()
    # legend.set_title(None)

    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()


    ax1.set_ylabel(y_label, fontsize=5)
    # ax1.set_ylabel(None)
    ax2.set_ylabel(None)
    ax3.set_ylabel(None)
    ax1.set_xlabel(None)
    ax2.set_xlabel(None)
    ax3.set_xlabel(None)

    ax2.tick_params(axis='y',   # Apply to the y-axis
                which='both',   # Both major and minor ticks
                left=False,     # No ticks on the left side
                labelleft=False # No tick labels on the left side
               )
    ax2.spines['left'].set_visible(False)  # Hide the left spine

    ax3.tick_params(axis='y',   # Apply to the y-axis
                which='both',   # Both major and minor ticks
                left=False,     # No ticks on the left side
                labelleft=False # No tick labels on the left side
               )
    ax3.spines['left'].set_visible(False)  # Hide the left spine

    if formersci_notation:
        # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(utils_vis.sci_notation_fmt))

        sf = ticker.ScalarFormatter(useMathText=True)
        sf.set_powerlimits((3, 3))
        ax1.yaxis.set_major_formatter(sf)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))

    if pairs_to_test is not None and annotate_stats:

        # Create an Annotator
        annotator = Annotator(
            ax1,
            pairs=pairs_to_test,
            data=df,
            x='group',
            y=feature,
            hue='trial_filter'
        )
        # Configure and apply the statistical test
        annotator.configure(
            # test='t-test_ind',
            test='Mann-Whitney',
            text_format='star',
            loc='outside',
            line_offset_to_group=10,
            line_offset=1,
            text_offset=0.1,
            line_height=0.03,
            hide_non_significant=True,
        )
        annotator.apply_and_annotate()

        ## Compute and print 95% CI for paired or unpaired data
        for pair in pairs_to_test:
            print(f"Generating 95% CI for pair: {pair}")
            group1 = df[(df['group'] == pair[0][0]) & (df['trial_filter'] == pair[0][1])][feature].values
            group2 = df[(df['group'] == pair[1][0]) & (df['trial_filter'] == pair[1][1])][feature].values

            if len(group1) > 1 and len(group2) > 1:
                if len(group1) == len(group2):
                    paired_95_CI(group1, group2)
                else:
                    unpaired_95_CI(group1, group2)

    # ax.margins(y=0.2)  # 20% margin above the data
    if y_max is not None:
        ax1.set_ylim(0, y_max)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05) # Bring the subplots closer together

    save_path = os.path.join(vis_dir, 'stats_plot_' + feature + '_' + suffix + '.pdf')

    fig.savefig(save_path, dpi=600, transparent=True, format='pdf', bbox_inches=None)


def print_stats(df: pd.DataFrame,
                agg_fn=np.median,
                prec: int = 3,
                *,
                split: bool = True,
                by_config: bool = False):
    """
    Pretty-print stats per session, then summary rows.

    by_config = False → one row per **(subject × array)**
    by_config = True  → one row per **(subject × cohort × array)**
    """

    # ────────────────────────── config ───────────────────────────
    GROUP_A   = {'ballistic', 'near'}
    GROUP_B   = {'sustained', 'far'}
    NUM_COLS  = ['peak_value', 'peak_time', 'peak_onset_time', 'peak_duration']


    # ───────────────────────── helpers ───────────────────────────
    def _meta(df):
        meta          = df['session'].str.extract(r'sub-(?P<subject>N\d+)_ses-(?P<date>\d{8})')
        meta['cohort'] = df['group'].str.split().str[-1]
        df             = df.join(meta)
        df['Session Info'] = df['subject'] + ' ' + df['date'] + ' ' + meta['cohort']
        df.rename(columns={'unit_filter': 'Array Location'}, inplace=True)
        return df


    def _fmt(x, prec):      return f"{x:.{prec}f}" if np.isfinite(x) else "–"


    def _agg_pair(frame, fn):
        gA = frame[frame['trial_filter'].isin(GROUP_A)][NUM_COLS].agg(fn)
        gB = frame[frame['trial_filter'].isin(GROUP_B)][NUM_COLS].agg(fn)
        return gA, gB


    df = _meta(df.copy())

    # ── quick row-builder ───────────────────────────
    def make_row(label, sub):
        if split:
            a, b  = _agg_pair(sub, agg_fn)
            j     = lambda m: f"{_fmt(a[m], prec)} / {_fmt(b[m], prec)}"
            note  = "ballistic+near / sustained+far"
        else:
            pooled = sub[NUM_COLS].agg(agg_fn)
            j      = lambda m: _fmt(pooled[m], prec)
            note   = "all conditions pooled"

        return [label,
                sub['Array Location'].iat[0] if 'Array Location' in sub else "",
                note,
                j('peak_value'),
                j('peak_time'),
                j('peak_onset_time'),
                j('peak_duration')]

    # ── table scaffold ──────────────────────────────
    tbl = PrettyTable()
    tbl.field_names = ["Session Info", "Array Location", "Conditions",
                       "Peak Magnitude", "Peak Time",
                       "Peak Onset Time", "Peak Duration"]

    # 1) session-wise rows
    for (sess, arr), sub in df.groupby(['Session Info', 'Array Location']):
        tbl.add_row(make_row(sess, sub))

    # 2) summary rows
    if by_config:                          # subject × cohort × array
        grp_cols = ['subject', 'cohort', 'Array Location']
        for (subj, coh, arr), g in df.groupby(grp_cols):
            tbl.add_row(make_row(f"{subj} {coh} {arr}", g))
    else:                                  # subject × array   (⇐ **NEW**)
        grp_cols = ['subject', 'Array Location']
        for (subj, arr), g in df.groupby(grp_cols):
            tbl.add_row(make_row(f"{subj} {arr} AVG", g))

    print(tbl)


def analyze_significance(df, feature, pairs_to_test, correction_factor=1):
    
    from prettytable import PrettyTable
    from scipy.stats import wilcoxon, mannwhitneyu, sem
    import numpy as np
    from math import isnan

    # Helper functions that return (mean_diff, ci_lower, ci_upper) instead of printing
    def compute_paired_95_CI(array1, array2):
        differences = array1 - array2
        mean_diff = np.mean(differences)
        sem_diff = sem(differences)  # = std(differences)/sqrt(n)
        df = len(differences) - 1
        if df < 1:
            return np.nan, np.nan, np.nan
        # t-crit for 95% CI (two-tailed)
        t_crit = t.ppf(1 - 0.025, df)
        margin_of_error = t_crit * sem_diff
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error
        return mean_diff, ci_lower, ci_upper

    def compute_unpaired_95_CI(array1, array2):
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)
        mean_diff = mean1 - mean2
        var1 = np.var(array1, ddof=1)
        var2 = np.var(array2, ddof=1)
        n1, n2 = len(array1), len(array2)
        if n1 < 2 or n2 < 2:
            return np.nan, np.nan, np.nan
        se_diff = np.sqrt(var1 / n1 + var2 / n2)
        # Welch–Satterthwaite DF
        df_numer = (var1/n1 + var2/n2)**2
        df_denom = (var1/n1)**2 / (n1 - 1) + (var2/n2)**2 / (n2 - 1)
        if df_denom == 0:
            return mean_diff, np.nan, np.nan
        df_eff = df_numer / df_denom
        t_crit = t.ppf(1 - 0.025, df_eff)
        margin = t_crit * se_diff
        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin
        return mean_diff, ci_lower, ci_upper

    # Helper function for p-value annotation
    def pval_annotation(p, correction_factor):
        """
        Annotation legend:
              ns: 5.00e-02 < p <= 1.00e+00
               *: 1.00e-02 < p <= 5.00e-02
              **: 1.00e-03 < p <= 1.00e-02
             ***: 1.00e-04 < p <= 1.00e-03
            ****: p <= 1.00e-04
        """
        p_ = p * correction_factor
        if p_ <= 1.0e-4:
            return "****"
        elif p_ <= 1.0e-3:
            return "***"
        elif p_ <= 1.0e-2:
            return "**"
        elif p_ <= 5.0e-2:
            return "*"
        else:
            return "ns"

    # Build the PrettyTable
    table = PrettyTable()
    table.field_names = [
        "Comparison",
        "# Trials (Grp1)",
        "# Trials (Grp2)",
        # f"Mean ± SEM ({feature}; Grp1)",
        # f"Mean ± SEM ({feature}; Grp2)",
        "Test",
        # "Statistic",
        "p-value",
        "p-val Annotation",
        "Mean Diff",
        "95% CI"
    ]

    for pair in pairs_to_test:
        (group1, cond1), (group2, cond2) = pair

        # Subset data for each group
        arr1 = df[(df["group"] == group1) & (df["trial_filter"] == cond1)][feature].dropna().values
        arr2 = df[(df["group"] == group2) & (df["trial_filter"] == cond2)][feature].dropna().values

        # Basic stats for each group
        mean1 = np.mean(arr1) if len(arr1) > 0 else np.nan
        mean2 = np.mean(arr2) if len(arr2) > 0 else np.nan
        sem1  = sem(arr1) if len(arr1) > 1 else 0.0
        sem2  = sem(arr2) if len(arr2) > 1 else 0.0

        # Decide paired or unpaired test
        if len(arr1) == len(arr2) and len(arr1) > 1:
        # if False:
            # Paired: Wilcoxon
            stat, p_val = wilcoxon(arr1, arr2)
            mean_diff, ci_lower, ci_upper = compute_paired_95_CI(arr1, arr2)
            test_name = "Wilcoxon (paired)"
        else:
            # Unpaired: Mann–Whitney
            if (len(arr1) > 0) and (len(arr2) > 0):
                stat, p_val = mannwhitneyu(arr1, arr2)
                mean_diff, ci_lower, ci_upper = compute_unpaired_95_CI(arr1, arr2)
                test_name = "Mann–Whitney U"
            else:
                # Edge case: not enough data
                stat, p_val = np.nan, np.nan
                mean_diff, ci_lower, ci_upper = np.nan, np.nan, np.nan
                test_name = "Mann–Whitney U (insufficient data)"

        # ipdb.set_trace()

        # Generate star annotation
        stars = pval_annotation(p_val if not isnan(p_val) else 1.0, correction_factor)

        # Build the row
        comp_label = f"{group1} {cond1} vs {group2} {cond2}"
        row = [
            comp_label,
            len(arr1),
            len(arr2),
            # f"{mean1:.3f} ± {sem1:.3f}",
            # f"{mean2:.3f} ± {sem2:.3f}",
            test_name,
            # f"{stat:.3f}",
            f"{p_val:.3g}",    # format p-value
            stars,
            f"{mean_diff:.3f}",
            f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        ]
        table.add_row(row)

    # Finally, print the table
    print(table)


def combined_plot_N1_N2_N2_RadialGrid(feature):

    df_N1 = construct_df(
        sessions=N1_sessions,
        trial_filters=['fast', 'slow'],
        unit_filters=['MC'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    df_N2 = construct_df(
        sessions=N2_sessions,
        trial_filters=['fast', 'slow'],
        unit_filters=['MC-LAT', 'MC-MED'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    df_N2_RadialGrid = construct_df(
        sessions=N2_RadialGrid_sessions,
        trial_filters=['near', 'far'],
        unit_filters=['MC-LAT', 'MC-MED'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    df_N2_CenterStartInterleave = construct_df(
        sessions=N2_CenterStartInterleave_sessions,
        trial_filters=['fast', 'slow'],
        unit_filters=['MC-LAT', 'MC-MED'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    # For N1
    # df_N1 = df_N1.copy()
    df_N1['group'] = df_N1.apply(lambda row: label_group(row, 'N1'), axis=1)
    df_N1['task'] = 'CenterStart'

    # For N2
    # df_N2 = df_N2.copy()
    df_N2['group'] = df_N2.apply(lambda row: label_group(row, 'N2'), axis=1)
    df_N2['task'] = 'CenterStart'

    # For N2 RadialGrid
    # df_N2_RadialGrid = df_N2_RadialGrid.copy()
    df_N2_RadialGrid['group'] = df_N2_RadialGrid.apply(lambda row: label_group(row, 'N2_RadialGrid'), axis=1)
    df_N2_RadialGrid['task'] = 'RadialGrid'

    # For N2 CenterStartInterleave
    # df_N2_CenterStartInterleave = df_N2_CenterStartInterleave.copy()
    df_N2_CenterStartInterleave['group'] = df_N2_CenterStartInterleave.apply(lambda row: label_group(row, 'N2_CenterStartInterleave'), axis=1)
    df_N2_CenterStartInterleave['task'] = 'CenterStartInterleave'

    df_combined = pd.concat(
        [df_N1, df_N2, df_N2_RadialGrid, df_N2_CenterStartInterleave], 
        ignore_index=True
    )

    df_combined.trial_filter = df_combined.trial_filter.replace('fast', 'ballistic')
    df_combined.trial_filter = df_combined.trial_filter.replace('slow', 'sustained')

    ipdb.set_trace()

    ## Filter out peak onset time or peak time larger than 1 second or less than 0.1 second
    df_combined = df_combined[(df_combined['peak_onset_time'] < 1.0) & (df_combined['peak_onset_time'] > 0.1)]
    df_combined = df_combined[(df_combined['peak_time'] < 1.0) & (df_combined['peak_time'] > 0.1)]

    pairs = [
        # --- ballistic vs. sustained (or near vs. far) ---
        (("N1 MC CO", "ballistic"),   ("N1 MC CO", "sustained")),
        (("N2 MCL CO", "ballistic"),  ("N2 MCL CO", "sustained")),
        (("N2 MCM CO", "ballistic"),  ("N2 MCM CO", "sustained")),
        (("N2 MCL COI", "ballistic"), ("N2 MCL COI", "sustained")),
        (("N2 MCM COI", "ballistic"), ("N2 MCM COI", "sustained")),
        (("N2 MCL RG", "near"),       ("N2 MCL RG", "far")),
        (("N2 MCM RG", "near"),       ("N2 MCM RG", "far")),

        # --- N1 vs. N2 (ballistic vs. ballistic, sustained vs. sustained) ---
        (("N1 MC CO", "ballistic"), ("N2 MCL CO", "ballistic")),
        (("N1 MC CO", "ballistic"), ("N2 MCM CO", "ballistic")),
        (("N1 MC CO", "sustained"), ("N2 MCL CO", "sustained")),
        (("N1 MC CO", "sustained"), ("N2 MCM CO", "sustained")),

        # --- N2 vs. N2 RadialGrid ---
        (("N2 MCL CO", "ballistic"), ("N2 MCL COI", "ballistic")),
        (("N2 MCL CO", "sustained"), ("N2 MCL COI", "sustained")),
        (("N2 MCM CO", "ballistic"), ("N2 MCM COI", "ballistic")),
        (("N2 MCM CO", "sustained"), ("N2 MCM COI", "sustained")),

        (("N2 MCL CO", "ballistic"), ("N2 MCL RG", "near")),
        (("N2 MCL CO", "sustained"), ("N2 MCL RG", "far")),
        (("N2 MCM CO", "ballistic"), ("N2 MCM RG", "near")),
        (("N2 MCM CO", "sustained"), ("N2 MCM RG", "far")),

        # --- N2 MCL vs. MCM ---
        (("N2 MCL CO", "ballistic"),  ("N2 MCM CO", "ballistic")),
        (("N2 MCL CO", "sustained"),  ("N2 MCM CO", "sustained")),
        (("N2 MCL COI", "ballistic"), ("N2 MCM COI", "ballistic")),
        (("N2 MCL COI", "sustained"), ("N2 MCM COI", "sustained")),
        (("N2 MCL RG", "near"),       ("N2 MCM RG", "near")),
        (("N2 MCL RG", "far"),        ("N2 MCM RG", "far")),
    ]

    n_pairs = len(pairs)

    if feature == 'peak_onset_time':
        plot_stats(
            df_combined, 
            feature='peak_onset_time', 
            # y_max=5, # With significance brackets
            y_max=1.0, # Without significance brackets
            pairs_to_test=pairs,
            annotate_stats=False,
            y_label='Peak Onset Time (s)',
            suffix='combined',
            show_x_labels=False)
    
    elif feature == 'peak_time':
        plot_stats(
            df_combined, 
            feature='peak_time',
            # y_max=5, # With significance brackets
            y_max=1.0, # Without significance brackets 
            pairs_to_test=pairs,
            annotate_stats=False,
            y_label='Peak Time (s)',
            suffix='combined',
            show_x_labels=False)

    elif feature == 'peak_duration':
        plot_stats(
            df_combined, 
            feature='peak_duration', 
            # y_max=0.6, # With significance brackets
            y_max=0.3, # Without significance brackets
            pairs_to_test=pairs,
            annotate_stats=False,
            y_label='Peak Duration (s)',
            suffix='combined',
            show_x_labels=True)

    elif feature == 'peak_value':
        plot_stats(
            df_combined, 
            feature='peak_value', 
            # y_max=50000, # With significance brackets
            y_max=25000, # Without significance brackets
            pairs_to_test=pairs,
            annotate_stats=False,
            y_label='Peak Magnitude (($\Delta$Hz / s)$^2$)',
            formersci_notation=True,
            suffix='combined',
            show_x_labels=True)
    
    
    print_stats(df_combined, split=True)
    analyze_significance(df_combined, feature=feature, pairs_to_test=pairs, correction_factor=n_pairs)
    

def combined_plot_N1_N2(feature):

    df_N1 = construct_df(
        sessions=N1_sessions,
        trial_filters=['fast', 'slow'],
        unit_filters=['MC'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    df_N2 = construct_df(
        sessions=N2_sessions,
        trial_filters=['fast', 'slow'],
        unit_filters=['MC-LAT', 'MC-MED'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    # For N1
    # df_N1 = df_N1.copy()
    df_N1['group'] = df_N1.apply(lambda row: label_group(row, 'N1'), axis=1)

    # For N2
    # df_N2 = df_N2.copy()
    df_N2['group'] = df_N2.apply(lambda row: label_group(row, 'N2'), axis=1)

    df_combined = pd.concat(
        [df_N1, df_N2], 
        ignore_index=True
    )

    pairs = [
        # --- ballistic vs. sustained (or near vs. far) ---
        (("N1 MC CO", "ballistic"), ("N1 MC CO", "sustained")),
        (("N2 MCL CO", "ballistic"), ("N2 MCL CO", "sustained")),
        (("N2 MCM CO", "ballistic"), ("N2 MCM CO", "sustained")),

        # --- N1 vs. N2 (ballistic vs. ballistic, sustained vs. sustained) ---
        (("N1 MC CO", "ballistic"), ("N2 MCL CO", "ballistic")),
        (("N1 MC CO", "ballistic"), ("N2 MCM CO", "ballistic")),
        (("N1 MC CO", "sustained"), ("N2 MCL CO", "sustained")),
        (("N1 MC CO", "sustained"), ("N2 MCM CO", "sustained")),

        # --- N2 MCL CO vs. MC-MED ---
        (("N2 MCL CO", "ballistic"), ("N2 MCM CO", "ballistic")),
        (("N2 MCL CO", "sustained"), ("N2 MCM CO", "sustained")),
    ]

    if feature == 'peak_onset_time':
        plot_stats(
            df_combined, 
            feature='peak_onset_time', 
            pairs_to_test=pairs,
            y_label='Peak Onset Time (s)',
            y_max=5,
            fontsize=5,
            suffix='N1_N2')
    elif feature == 'peak_duration':
        plot_stats(
            df_combined, 
            feature='peak_duration', 
            pairs_to_test=pairs,
            y_label='Peak Duration (s)',
            y_max=0.6,
            fontsize=5,
            suffix='N1_N2')
    elif feature == 'peak_value':
        plot_stats(
            df_combined, 
            feature='peak_value', 
            pairs_to_test=pairs,
            y_label='Peak Magnitude (($\Delta$Hz / s)$^2$)',
            y_max=50000,
            fontsize=5,
            formersci_notation=True,
            suffix='N1_N2')
    elif feature == 'peak_time':
        plot_stats(
            df_combined, 
            feature='peak_time', 
            pairs_to_test=pairs,
            y_label='Peak Time (s)',
            y_max=5,
            fontsize=5,
            suffix='N1_N2')
        

# ────────────────────────────────────────────────────────────────
# utilities (reuse earlier variants)
# ────────────────────────────────────────────────────────────────
def cohen_d(x, y):
    n1, n2   = len(x), len(y)
    s1, s2   = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pool   = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / s_pool


def mean_diff_ci(x, y, alpha=0.05):
    """95 % CI of the *unpaired* mean difference using Welch SE."""
    mean1, mean2 = np.mean(x), np.mean(y)
    var1,  var2  = np.var(x, ddof=1), np.var(y, ddof=1)
    n1,   n2     = len(x), len(y)

    se    = np.sqrt(var1/n1 + var2/n2)
    df_eff = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    tcrit = t.ppf(1 - alpha/2, df_eff)
    margin = tcrit * se
    return (mean1 - mean2) - margin, (mean1 - mean2) + margin
# ────────────────────────────────────────────────────────────────


def answer_peak_questions():

    # ---------- build CO-only dataframe (arrays pooled) ----------
    df_N1 = construct_df(
        sessions=N1_sessions, trial_filters=['fast','slow'],
        unit_filters=['MC'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )
    df_N2 = construct_df(
        sessions=N2_sessions, trial_filters=['fast','slow'],
        unit_filters=['MC-LAT','MC-MED'],
        window_config='gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10',
        data_format='truncate_end',
    )

    df_N1['group'] = 'N1 CO'
    df_N2['group'] = 'N2 CO'
    df = pd.concat([df_N1, df_N2], ignore_index=True)

    df.trial_filter = df.trial_filter.replace({'fast':'ballistic', 'slow':'sustained'})
    df = df[(df.peak_onset_time.between(0.1,1.0)) & (df.peak_time.between(0.1,1.0))]

    def pick(g, c, col):
        return df[(df.group==g) & (df.trial_filter==c)][col].dropna().values

    # ── Q1  ────────────────────────────────────────────────────
    print("\n─ Q1 · Initial PEAK-ONSET times  (ballistic vs sustained) ─")
    for subj in ['N1 CO', 'N2 CO']:
        a = pick(subj, 'ballistic', 'peak_onset_time')
        b = pick(subj, 'sustained', 'peak_onset_time')

        d         = cohen_d(a, b)
        lo, hi    = mean_diff_ci(a, b)
        stat, p   = mannwhitneyu(a, b)

        print(f"{subj:<6}  Δmean = {np.mean(a)-np.mean(b):+.3f} s "
              f"[95 % CI {lo:+.3f},{hi:+.3f}]   "
              f"Cohen d {d:+.2f}   p={p:.3g}   n={len(a)}/{len(b)}")

    # ── Q2  ────────────────────────────────────────────────────
    print("\n─ Q2 · PEAK-TIME difference  (N1 vs N2, arrays pooled) ─")
    for cond in ['ballistic','sustained']:
        a = pick('N1 CO', cond, 'peak_time')
        b = pick('N2 CO', cond, 'peak_time')

        d       = cohen_d(a, b)
        lo, hi  = mean_diff_ci(a, b)
        stat, p = mannwhitneyu(a, b)

        print(f"{cond:<10}  Δmean = {np.mean(a)-np.mean(b):+.3f} s "
              f"[95 % CI {lo:+.3f},{hi:+.3f}]   "
              f"Cohen d {d:+.2f}   p={p:.3g}   n={len(a)}/{len(b)}")

    # optional pooled-all-trials row
    a_all = df[df.group=='N1 CO'].peak_time.values
    b_all = df[df.group=='N2 CO'].peak_time.values
    d_all = cohen_d(a_all, b_all)
    lo, hi = mean_diff_ci(a_all, b_all)
    stat, p = mannwhitneyu(a_all, b_all)
    print(f"all trials  Δmean = {np.mean(a_all)-np.mean(b_all):+.3f} s "
          f"[95 % CI {lo:+.3f},{hi:+.3f}]   "
          f"Cohen d {d_all:+.2f}   p={p:.3g}   n={len(a_all)}/{len(b_all)}")
# ────────────────────────────────────────────────────────────────



if __name__ == "__main__":

    answer_peak_questions()

    # combined_plot_N1_N2_N2_RadialGrid(feature='peak_value')
    # combined_plot_N1_N2_N2_RadialGrid(feature='peak_time')
    # combined_plot_N1_N2_N2_RadialGrid(feature='peak_onset_time')
    # combined_plot_N1_N2_N2_RadialGrid(feature='peak_duration')
    
    # combined_plot_N1_N2(feature='peak_onset_time')
    # combined_plot_N1_N2(feature='peak_duration')
    # combined_plot_N1_N2(feature='peak_value')
    # combined_plot_N1_N2(feature='peak_time')
    

# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------------+
# |                  Comparison                  | # Trials (Grp1) | # Trials (Grp2) |      Test      | p-value  | p-val Annotation | Mean Diff |         95% CI         |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------------+
# |   N1 MC CO ballistic vs N1 MC CO sustained   |       151       |       173       | Mann–Whitney U |  0.0055  |        ns        |  849.062  |  [317.976, 1380.148]   |
# |  N2 MCL CO ballistic vs N2 MCL CO sustained  |       423       |       403       | Mann–Whitney U | 1.52e-05 |       ***        |  782.752  |  [394.268, 1171.237]   |
# |  N2 MCM CO ballistic vs N2 MCM CO sustained  |       417       |       371       | Mann–Whitney U |  0.048   |        ns        |  -40.652  |  [-363.772, 282.468]   |
# | N2 MCL COI ballistic vs N2 MCL COI sustained |       473       |       360       | Mann–Whitney U | 5.24e-08 |       ****       | -1001.738 | [-1411.689, -591.787]  |
# | N2 MCM COI ballistic vs N2 MCM COI sustained |       472       |       355       | Mann–Whitney U |   0.13   |        ns        |  -356.334 |   [-803.662, 90.994]   |
# |       N2 MCL RG near vs N2 MCL RG far        |       250       |       253       | Mann–Whitney U |  0.147   |        ns        |  -194.681 |  [-591.745, 202.383]   |
# |       N2 MCM RG near vs N2 MCM RG far        |       241       |       249       | Mann–Whitney U |  0.457   |        ns        |  370.090  |  [-189.396, 929.576]   |
# |  N1 MC CO ballistic vs N2 MCL CO ballistic   |       151       |       423       | Mann–Whitney U |  0.636   |        ns        |   26.174  |  [-487.480, 539.829]   |
# |  N1 MC CO ballistic vs N2 MCM CO ballistic   |       151       |       417       | Mann–Whitney U | 6.62e-18 |       ****       |  1881.021 |  [1384.196, 2377.845]  |
# |  N1 MC CO sustained vs N2 MCL CO sustained   |       173       |       403       | Mann–Whitney U |  0.331   |        ns        |  -40.135  |  [-451.788, 371.517]   |
# |  N1 MC CO sustained vs N2 MCM CO sustained   |       173       |       371       | Mann–Whitney U | 3.1e-09  |       ****       |  991.307  |  [616.826, 1365.789]   |
# | N2 MCL CO ballistic vs N2 MCL COI ballistic  |       423       |       473       | Mann–Whitney U | 7.31e-06 |       ***        |  -824.485 | [-1218.159, -430.811]  |
# | N2 MCL CO sustained vs N2 MCL COI sustained  |       403       |       360       | Mann–Whitney U | 6.02e-37 |       ****       | -2608.975 | [-3013.947, -2204.004] |
# | N2 MCM CO ballistic vs N2 MCM COI ballistic  |       417       |       472       | Mann–Whitney U | 8.1e-27  |       ****       | -1876.619 | [-2258.869, -1494.370] |
# | N2 MCM CO sustained vs N2 MCM COI sustained  |       371       |       355       | Mann–Whitney U | 7.55e-25 |       ****       | -2192.302 | [-2590.402, -1794.202] |
# |    N2 MCL CO ballistic vs N2 MCL RG near     |       423       |       250       | Mann–Whitney U | 0.00505  |        ns        |  755.487  |  [343.087, 1167.886]   |
# |     N2 MCL CO sustained vs N2 MCL RG far     |       403       |       253       | Mann–Whitney U | 0.00335  |        ns        |  -221.947 |  [-594.101, 150.207]   |
# |    N2 MCM CO ballistic vs N2 MCM RG near     |       417       |       241       | Mann–Whitney U | 8.84e-26 |       ****       | -2334.069 | [-2837.365, -1830.773] |
# |     N2 MCM CO sustained vs N2 MCM RG far     |       371       |       249       | Mann–Whitney U | 4.88e-22 |       ****       | -1923.327 | [-2329.107, -1517.548] |
# |  N2 MCL CO ballistic vs N2 MCM CO ballistic  |       423       |       417       | Mann–Whitney U | 7.03e-28 |       ****       |  1854.846 |  [1476.958, 2232.735]  |
# |  N2 MCL CO sustained vs N2 MCM CO sustained  |       403       |       371       | Mann–Whitney U | 4.78e-09 |       ****       |  1031.442 |  [695.969, 1366.916]   |
# | N2 MCL COI ballistic vs N2 MCM COI ballistic |       473       |       472       | Mann–Whitney U | 4.01e-07 |       ****       |  802.712  |  [404.849, 1200.575]   |
# | N2 MCL COI sustained vs N2 MCM COI sustained |       360       |       355       | Mann–Whitney U | 4.14e-13 |       ****       |  1448.116 |  [990.039, 1906.192]   |
# |       N2 MCL RG near vs N2 MCM RG near       |       250       |       241       | Mann–Whitney U | 0.000199 |        **        | -1234.709 | [-1764.209, -705.210]  |
# |        N2 MCL RG far vs N2 MCM RG far        |       253       |       249       | Mann–Whitney U |  0.0767  |        ns        |  -669.938 | [-1106.386, -233.490]  |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------------+


# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |                  Comparison                  | # Trials (Grp1) | # Trials (Grp2) |      Test      | p-value  | p-val Annotation | Mean Diff |      95% CI      |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |   N1 MC CO ballistic vs N1 MC CO sustained   |       151       |       173       | Mann–Whitney U | 2.46e-09 |       ****       |   -0.025  | [-0.034, -0.016] |
# |  N2 MCL CO ballistic vs N2 MCL CO sustained  |       423       |       403       | Mann–Whitney U |  0.0264  |        ns        |   -0.017  | [-0.029, -0.004] |
# |  N2 MCM CO ballistic vs N2 MCM CO sustained  |       417       |       371       | Mann–Whitney U |  0.298   |        ns        |   -0.022  | [-0.038, -0.005] |
# | N2 MCL COI ballistic vs N2 MCL COI sustained |       473       |       360       | Mann–Whitney U | 3.52e-06 |       ****       |   -0.017  | [-0.030, -0.004] |
# | N2 MCM COI ballistic vs N2 MCM COI sustained |       472       |       355       | Mann–Whitney U |  0.553   |        ns        |   -0.009  | [-0.021, 0.002]  |
# |       N2 MCL RG near vs N2 MCL RG far        |       250       |       253       | Mann–Whitney U | 0.00136  |        *         |   -0.023  | [-0.036, -0.011] |
# |       N2 MCM RG near vs N2 MCM RG far        |       241       |       249       | Mann–Whitney U | 0.00365  |        ns        |   -0.035  | [-0.055, -0.016] |
# |  N1 MC CO ballistic vs N2 MCL CO ballistic   |       151       |       423       | Mann–Whitney U | 4.39e-73 |       ****       |   -0.177  | [-0.185, -0.170] |
# |  N1 MC CO ballistic vs N2 MCM CO ballistic   |       151       |       417       | Mann–Whitney U | 2.38e-70 |       ****       |   -0.172  | [-0.181, -0.162] |
# |  N1 MC CO sustained vs N2 MCL CO sustained   |       173       |       403       | Mann–Whitney U | 1.67e-68 |       ****       |   -0.169  | [-0.182, -0.156] |
# |  N1 MC CO sustained vs N2 MCM CO sustained   |       173       |       371       | Mann–Whitney U | 1.77e-61 |       ****       |   -0.168  | [-0.184, -0.152] |
# | N2 MCL CO ballistic vs N2 MCL COI ballistic  |       423       |       473       | Mann–Whitney U | 1.9e-08  |       ****       |   0.004   | [-0.006, 0.015]  |
# | N2 MCL CO sustained vs N2 MCL COI sustained  |       403       |       360       | Mann–Whitney U |  0.111   |        ns        |   0.005   | [-0.010, 0.019]  |
# | N2 MCM CO ballistic vs N2 MCM COI ballistic  |       417       |       472       | Mann–Whitney U | 4.3e-08  |       ****       |   0.030   |  [0.019, 0.040]  |
# | N2 MCM CO sustained vs N2 MCM COI sustained  |       371       |       355       | Mann–Whitney U | 4.72e-06 |       ***        |   0.042   |  [0.025, 0.059]  |
# |    N2 MCL CO ballistic vs N2 MCL RG near     |       423       |       250       | Mann–Whitney U |  0.277   |        ns        |   0.009   |  [0.000, 0.019]  |
# |     N2 MCL CO sustained vs N2 MCL RG far     |       403       |       253       | Mann–Whitney U |  0.677   |        ns        |   0.002   | [-0.013, 0.018]  |
# |    N2 MCM CO ballistic vs N2 MCM RG near     |       417       |       241       | Mann–Whitney U |  0.0164  |        ns        |   0.020   |  [0.007, 0.033]  |
# |     N2 MCM CO sustained vs N2 MCM RG far     |       371       |       249       | Mann–Whitney U |  0.866   |        ns        |   0.006   | [-0.015, 0.028]  |
# |  N2 MCL CO ballistic vs N2 MCM CO ballistic  |       423       |       417       | Mann–Whitney U | 3.18e-06 |       ****       |   0.006   | [-0.005, 0.016]  |
# |  N2 MCL CO sustained vs N2 MCM CO sustained  |       403       |       371       | Mann–Whitney U | 0.000446 |        *         |   0.001   | [-0.017, 0.019]  |
# | N2 MCL COI ballistic vs N2 MCM COI ballistic |       473       |       472       | Mann–Whitney U | 5.12e-08 |       ****       |   0.031   |  [0.020, 0.041]  |
# | N2 MCL COI sustained vs N2 MCM COI sustained |       360       |       355       | Mann–Whitney U | 5.05e-16 |       ****       |   0.038   |  [0.024, 0.052]  |
# |       N2 MCL RG near vs N2 MCM RG near       |       250       |       241       | Mann–Whitney U | 3.28e-07 |       ****       |   0.017   |  [0.004, 0.029]  |
# |        N2 MCL RG far vs N2 MCM RG far        |       253       |       249       | Mann–Whitney U | 1.35e-05 |       ***        |   0.005   | [-0.015, 0.024]  |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+


# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |                  Comparison                  | # Trials (Grp1) | # Trials (Grp2) |      Test      | p-value  | p-val Annotation | Mean Diff |      95% CI      |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |   N1 MC CO ballistic vs N1 MC CO sustained   |       151       |       173       | Mann–Whitney U | 1.54e-05 |       ***        |   -0.017  | [-0.026, -0.008] |
# |  N2 MCL CO ballistic vs N2 MCL CO sustained  |       423       |       403       | Mann–Whitney U | 0.00326  |        ns        |   -0.020  | [-0.034, -0.007] |
# |  N2 MCM CO ballistic vs N2 MCM CO sustained  |       417       |       371       | Mann–Whitney U |  0.323   |        ns        |   -0.023  | [-0.040, -0.005] |
# | N2 MCL COI ballistic vs N2 MCL COI sustained |       473       |       360       | Mann–Whitney U |  0.0307  |        ns        |   -0.011  | [-0.025, 0.003]  |
# | N2 MCM COI ballistic vs N2 MCM COI sustained |       472       |       355       | Mann–Whitney U |  0.648   |        ns        |   -0.008  | [-0.021, 0.005]  |
# |       N2 MCL RG near vs N2 MCL RG far        |       250       |       253       | Mann–Whitney U | 0.000612 |        *         |   -0.027  | [-0.041, -0.012] |
# |       N2 MCM RG near vs N2 MCM RG far        |       241       |       249       | Mann–Whitney U | 0.00548  |        ns        |   -0.039  | [-0.059, -0.018] |
# |  N1 MC CO ballistic vs N2 MCL CO ballistic   |       151       |       423       | Mann–Whitney U | 3.22e-71 |       ****       |   -0.159  | [-0.168, -0.150] |
# |  N1 MC CO ballistic vs N2 MCM CO ballistic   |       151       |       417       | Mann–Whitney U | 7.01e-71 |       ****       |   -0.176  | [-0.186, -0.165] |
# |  N1 MC CO sustained vs N2 MCL CO sustained   |       173       |       403       | Mann–Whitney U | 1.39e-68 |       ****       |   -0.162  | [-0.176, -0.148] |
# |  N1 MC CO sustained vs N2 MCM CO sustained   |       173       |       371       | Mann–Whitney U | 2.3e-67  |       ****       |   -0.182  | [-0.198, -0.165] |
# | N2 MCL CO ballistic vs N2 MCL COI ballistic  |       423       |       473       | Mann–Whitney U |  0.679   |        ns        |   -0.010  | [-0.022, 0.002]  |
# | N2 MCL CO sustained vs N2 MCL COI sustained  |       403       |       360       | Mann–Whitney U |  0.996   |        ns        |   -0.001  | [-0.017, 0.016]  |
# | N2 MCM CO ballistic vs N2 MCM COI ballistic  |       417       |       472       | Mann–Whitney U | 2.99e-13 |       ****       |   0.042   |  [0.030, 0.053]  |
# | N2 MCM CO sustained vs N2 MCM COI sustained  |       371       |       355       | Mann–Whitney U | 3.38e-12 |       ****       |   0.056   |  [0.038, 0.074]  |
# |    N2 MCL CO ballistic vs N2 MCL RG near     |       423       |       250       | Mann–Whitney U |  0.886   |        ns        |   0.008   | [-0.003, 0.018]  |
# |     N2 MCL CO sustained vs N2 MCL RG far     |       403       |       253       | Mann–Whitney U |  0.318   |        ns        |   0.001   | [-0.016, 0.018]  |
# |    N2 MCM CO ballistic vs N2 MCM RG near     |       417       |       241       | Mann–Whitney U | 0.000182 |        **        |   0.028   |  [0.013, 0.042]  |
# |     N2 MCM CO sustained vs N2 MCM RG far     |       371       |       249       | Mann–Whitney U |   0.28   |        ns        |   0.012   | [-0.011, 0.035]  |
# |  N2 MCL CO ballistic vs N2 MCM CO ballistic  |       423       |       417       | Mann–Whitney U |  0.0764  |        ns        |   -0.017  | [-0.029, -0.006] |
# |  N2 MCL CO sustained vs N2 MCM CO sustained  |       403       |       371       | Mann–Whitney U |  0.786   |        ns        |   -0.019  | [-0.038, -0.001] |
# | N2 MCL COI ballistic vs N2 MCM COI ballistic |       473       |       472       | Mann–Whitney U | 8.38e-08 |       ****       |   0.034   |  [0.023, 0.046]  |
# | N2 MCL COI sustained vs N2 MCM COI sustained |       360       |       355       | Mann–Whitney U | 2.38e-11 |       ****       |   0.037   |  [0.022, 0.053]  |
# |       N2 MCL RG near vs N2 MCM RG near       |       250       |       241       | Mann–Whitney U |  0.0886  |        ns        |   0.003   | [-0.011, 0.017]  |
# |        N2 MCL RG far vs N2 MCM RG far        |       253       |       249       | Mann–Whitney U |  0.119   |        ns        |   -0.009  | [-0.030, 0.012]  |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+


# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |                  Comparison                  | # Trials (Grp1) | # Trials (Grp2) |      Test      | p-value  | p-val Annotation | Mean Diff |      95% CI      |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+
# |   N1 MC CO ballistic vs N1 MC CO sustained   |       151       |       173       | Mann–Whitney U | 0.00535  |        ns        |   -0.006  | [-0.011, -0.001] |
# |  N2 MCL CO ballistic vs N2 MCL CO sustained  |       423       |       403       | Mann–Whitney U |   0.37   |        ns        |   0.001   | [-0.002, 0.005]  |
# |  N2 MCM CO ballistic vs N2 MCM CO sustained  |       417       |       371       | Mann–Whitney U |  0.0388  |        ns        |   0.002   | [-0.002, 0.005]  |
# | N2 MCL COI ballistic vs N2 MCL COI sustained |       473       |       360       | Mann–Whitney U |  0.314   |        ns        |   -0.001  | [-0.004, 0.002]  |
# | N2 MCM COI ballistic vs N2 MCM COI sustained |       472       |       355       | Mann–Whitney U |  0.154   |        ns        |   -0.002  | [-0.006, 0.001]  |
# |       N2 MCL RG near vs N2 MCL RG far        |       250       |       253       | Mann–Whitney U |  0.012   |        ns        |   -0.006  | [-0.010, -0.001] |
# |       N2 MCM RG near vs N2 MCM RG far        |       241       |       249       | Mann–Whitney U |  0.272   |        ns        |   0.002   | [-0.003, 0.007]  |
# |  N1 MC CO ballistic vs N2 MCL CO ballistic   |       151       |       423       | Mann–Whitney U | 0.000723 |        *         |   -0.006  | [-0.011, -0.002] |
# |  N1 MC CO ballistic vs N2 MCM CO ballistic   |       151       |       417       | Mann–Whitney U |  0.0613  |        ns        |   0.004   | [-0.001, 0.008]  |
# |  N1 MC CO sustained vs N2 MCL CO sustained   |       173       |       403       | Mann–Whitney U |   0.59   |        ns        |   0.001   | [-0.003, 0.005]  |
# |  N1 MC CO sustained vs N2 MCM CO sustained   |       173       |       371       | Mann–Whitney U | 1.97e-10 |       ****       |   0.011   |  [0.007, 0.016]  |
# | N2 MCL CO ballistic vs N2 MCL COI ballistic  |       423       |       473       | Mann–Whitney U | 1.23e-05 |       ***        |   0.006   |  [0.003, 0.009]  |
# | N2 MCL CO sustained vs N2 MCL COI sustained  |       403       |       360       | Mann–Whitney U |  0.0492  |        ns        |   0.003   |  [0.000, 0.007]  |
# | N2 MCM CO ballistic vs N2 MCM COI ballistic  |       417       |       472       | Mann–Whitney U | 1.81e-07 |       ****       |   -0.008  | [-0.011, -0.004] |
# | N2 MCM CO sustained vs N2 MCM COI sustained  |       371       |       355       | Mann–Whitney U | 4.93e-14 |       ****       |   -0.012  | [-0.015, -0.008] |
# |    N2 MCL CO ballistic vs N2 MCL RG near     |       423       |       250       | Mann–Whitney U |  0.312   |        ns        |   0.001   | [-0.003, 0.005]  |
# |     N2 MCL CO sustained vs N2 MCL RG far     |       403       |       253       | Mann–Whitney U |  0.0152  |        ns        |   -0.006  | [-0.010, -0.001] |
# |    N2 MCM CO ballistic vs N2 MCM RG near     |       417       |       241       | Mann–Whitney U |  0.0778  |        ns        |   -0.004  | [-0.008, 0.000]  |
# |     N2 MCM CO sustained vs N2 MCM RG far     |       371       |       249       | Mann–Whitney U |  0.0173  |        ns        |   -0.004  | [-0.008, 0.000]  |
# |  N2 MCL CO ballistic vs N2 MCM CO ballistic  |       423       |       417       | Mann–Whitney U | 5.1e-11  |       ****       |   0.010   |  [0.006, 0.013]  |
# |  N2 MCL CO sustained vs N2 MCM CO sustained  |       403       |       371       | Mann–Whitney U | 3.07e-12 |       ****       |   0.010   |  [0.007, 0.013]  |
# | N2 MCL COI ballistic vs N2 MCM COI ballistic |       473       |       472       | Mann–Whitney U |  0.0085  |        ns        |   -0.004  | [-0.007, -0.001] |
# | N2 MCL COI sustained vs N2 MCM COI sustained |       360       |       355       | Mann–Whitney U | 0.00731  |        ns        |   -0.005  | [-0.009, -0.001] |
# |       N2 MCL RG near vs N2 MCM RG near       |       250       |       241       | Mann–Whitney U | 0.00911  |        ns        |   0.005   | [-0.000, 0.009]  |
# |        N2 MCL RG far vs N2 MCM RG far        |       253       |       249       | Mann–Whitney U | 6.11e-09 |       ****       |   0.012   |  [0.007, 0.017]  |
# +----------------------------------------------+-----------------+-----------------+----------------+----------+------------------+-----------+------------------+


# +-----------------+----------------+--------------------------------+---------------------+---------------+-----------------+---------------+
# |   Session Info  | Array Location |           Conditions           |    Peak Magnitude   |   Peak Time   | Peak Onset Time | Peak Duration |
# +-----------------+----------------+--------------------------------+---------------------+---------------+-----------------+---------------+
# |  N1 20190412 CO |       MC       | ballistic+near / sustained+far | 7781.359 / 6459.513 | 0.173 / 0.196 |  0.126 / 0.135  | 0.056 / 0.058 |
# |  N1 20190517 CO |       MC       | ballistic+near / sustained+far | 5804.063 / 4174.614 | 0.174 / 0.195 |  0.128 / 0.148  | 0.050 / 0.060 |
# |  N1 20190528 CO |       MC       | ballistic+near / sustained+far | 4167.045 / 3873.012 | 0.192 / 0.203 |  0.149 / 0.155  | 0.058 / 0.067 |
# |  N2 20240516 CO |     MC-LAT     | ballistic+near / sustained+far | 8870.450 / 8502.879 | 0.329 / 0.314 |  0.264 / 0.264  | 0.064 / 0.064 |
# |  N2 20240516 CO |     MC-MED     | ballistic+near / sustained+far | 5994.398 / 5165.513 | 0.304 / 0.311 |  0.272 / 0.264  | 0.053 / 0.057 |
# |  N2 20240530 CO |     MC-LAT     | ballistic+near / sustained+far | 7741.312 / 4253.177 | 0.359 / 0.372 |  0.296 / 0.315  | 0.053 / 0.051 |
# |  N2 20240530 CO |     MC-MED     | ballistic+near / sustained+far | 8008.725 / 5958.091 | 0.318 / 0.316 |  0.275 / 0.286  | 0.056 / 0.047 |
# |  N2 20240816 CO |     MC-LAT     | ballistic+near / sustained+far | 6615.840 / 6077.524 | 0.351 / 0.358 |  0.266 / 0.279  | 0.067 / 0.067 |
# |  N2 20240816 CO |     MC-MED     | ballistic+near / sustained+far | 2753.055 / 2853.711 | 0.309 / 0.437 |  0.270 / 0.400  | 0.053 / 0.052 |
# |  N2 20240820 CO |     MC-LAT     | ballistic+near / sustained+far | 4450.110 / 3367.897 | 0.326 / 0.333 |  0.260 / 0.295  | 0.061 / 0.053 |
# |  N2 20240820 CO |     MC-MED     | ballistic+near / sustained+far | 2869.859 / 2324.496 | 0.327 / 0.370 |  0.292 / 0.348  | 0.043 / 0.037 |
# |  N2 20241015 CO |     MC-LAT     | ballistic+near / sustained+far | 3998.722 / 3990.193 | 0.353 / 0.376 |  0.290 / 0.308  | 0.062 / 0.066 |
# |  N2 20241015 CO |     MC-MED     | ballistic+near / sustained+far | 3544.581 / 4390.851 | 0.337 / 0.322 |  0.288 / 0.275  | 0.060 / 0.055 |
# |  N2 20241022 CO |     MC-LAT     | ballistic+near / sustained+far | 3594.970 / 3166.610 | 0.365 / 0.382 |  0.321 / 0.320  | 0.052 / 0.049 |
# |  N2 20241022 CO |     MC-MED     | ballistic+near / sustained+far | 2199.439 / 2774.069 | 0.358 / 0.333 |  0.338 / 0.292  | 0.043 / 0.040 |
# |  N2 20241105 RG |     MC-LAT     | ballistic+near / sustained+far | 3148.722 / 4192.182 | 0.337 / 0.339 |  0.285 / 0.282  | 0.059 / 0.061 |
# |  N2 20241105 RG |     MC-MED     | ballistic+near / sustained+far | 3977.974 / 4008.969 | 0.332 / 0.361 |  0.293 / 0.325  | 0.050 / 0.043 |
# |  N2 20241211 RG |     MC-LAT     | ballistic+near / sustained+far | 5394.919 / 6217.943 | 0.333 / 0.377 |  0.294 / 0.329  | 0.055 / 0.061 |
# |  N2 20241211 RG |     MC-MED     | ballistic+near / sustained+far | 5953.062 / 4998.364 | 0.316 / 0.338 |  0.296 / 0.312  | 0.048 / 0.057 |
# |  N2 20250408 RG |     MC-LAT     | ballistic+near / sustained+far | 5497.145 / 5403.012 | 0.366 / 0.364 |  0.291 / 0.290  | 0.063 / 0.072 |
# |  N2 20250408 RG |     MC-MED     | ballistic+near / sustained+far | 6541.712 / 6900.555 | 0.304 / 0.308 |  0.241 / 0.246  | 0.062 / 0.059 |
# | N2 20250417 COI |     MC-LAT     | ballistic+near / sustained+far | 5085.750 / 7717.144 | 0.346 / 0.360 |  0.309 / 0.318  | 0.053 / 0.053 |
# | N2 20250417 COI |     MC-MED     | ballistic+near / sustained+far | 5913.332 / 5299.265 | 0.308 / 0.318 |  0.256 / 0.259  | 0.057 / 0.059 |
# | N2 20250422 COI |     MC-LAT     | ballistic+near / sustained+far | 6635.530 / 6568.098 | 0.339 / 0.361 |  0.284 / 0.316  | 0.053 / 0.055 |
# | N2 20250422 COI |     MC-MED     | ballistic+near / sustained+far | 3793.014 / 3824.938 | 0.337 / 0.351 |  0.285 / 0.291  | 0.057 / 0.057 |
# | N2 20250509 COI |     MC-LAT     | ballistic+near / sustained+far | 6924.908 / 7857.621 | 0.305 / 0.323 |  0.247 / 0.258  | 0.057 / 0.064 |
# | N2 20250509 COI |     MC-MED     | ballistic+near / sustained+far | 6890.679 / 7943.535 | 0.285 / 0.296 |  0.234 / 0.242  | 0.063 / 0.065 |
# |    N1 MC AVG    |       MC       | ballistic+near / sustained+far | 5255.606 / 4697.478 | 0.178 / 0.197 |  0.132 / 0.148  | 0.055 / 0.061 |
# |  N2 MC-LAT AVG  |     MC-LAT     | ballistic+near / sustained+far | 5573.560 / 5534.905 | 0.342 / 0.356 |  0.283 / 0.297  | 0.058 / 0.059 |
# |  N2 MC-MED AVG  |     MC-MED     | ballistic+near / sustained+far | 4475.431 / 4589.678 | 0.318 / 0.325 |  0.274 / 0.279  | 0.055 / 0.054 |
# +-----------------+----------------+--------------------------------+---------------------+---------------+-----------------+---------------+

# +-----------------+----------------+-----------------------+----------------+-----------+-----------------+---------------+
# |   Session Info  | Array Location |       Conditions      | Peak Magnitude | Peak Time | Peak Onset Time | Peak Duration |
# +-----------------+----------------+-----------------------+----------------+-----------+-----------------+---------------+
# |  N1 20190412 CO |       MC       | all conditions pooled |    7160.770    |   0.180   |      0.132      |     0.056     |
# |  N1 20190517 CO |       MC       | all conditions pooled |    4917.206    |   0.185   |      0.138      |     0.055     |
# |  N1 20190528 CO |       MC       | all conditions pooled |    4030.888    |   0.198   |      0.152      |     0.063     |
# |  N2 20240516 CO |     MC-LAT     | all conditions pooled |    8711.082    |   0.319   |      0.264      |     0.064     |
# |  N2 20240516 CO |     MC-MED     | all conditions pooled |    5450.882    |   0.307   |      0.269      |     0.054     |
# |  N2 20240530 CO |     MC-LAT     | all conditions pooled |    5204.920    |   0.366   |      0.307      |     0.052     |
# |  N2 20240530 CO |     MC-MED     | all conditions pooled |    6702.757    |   0.317   |      0.279      |     0.052     |
# |  N2 20240816 CO |     MC-LAT     | all conditions pooled |    6378.134    |   0.353   |      0.273      |     0.067     |
# |  N2 20240816 CO |     MC-MED     | all conditions pooled |    2812.484    |   0.374   |      0.309      |     0.052     |
# |  N2 20240820 CO |     MC-LAT     | all conditions pooled |    4087.885    |   0.327   |      0.273      |     0.057     |
# |  N2 20240820 CO |     MC-MED     | all conditions pooled |    2595.315    |   0.332   |      0.296      |     0.041     |
# |  N2 20241015 CO |     MC-LAT     | all conditions pooled |    3996.934    |   0.367   |      0.297      |     0.064     |
# |  N2 20241015 CO |     MC-MED     | all conditions pooled |    3906.847    |   0.327   |      0.282      |     0.057     |
# |  N2 20241022 CO |     MC-LAT     | all conditions pooled |    3284.038    |   0.372   |      0.320      |     0.051     |
# |  N2 20241022 CO |     MC-MED     | all conditions pooled |    2453.047    |   0.352   |      0.319      |     0.042     |
# |  N2 20241105 RG |     MC-LAT     | all conditions pooled |    3745.044    |   0.338   |      0.283      |     0.060     |
# |  N2 20241105 RG |     MC-MED     | all conditions pooled |    3977.974    |   0.349   |      0.310      |     0.047     |
# |  N2 20241211 RG |     MC-LAT     | all conditions pooled |    5821.013    |   0.359   |      0.307      |     0.057     |
# |  N2 20241211 RG |     MC-MED     | all conditions pooled |    5430.398    |   0.333   |      0.307      |     0.051     |
# |  N2 20250408 RG |     MC-LAT     | all conditions pooled |    5437.342    |   0.364   |      0.290      |     0.066     |
# |  N2 20250408 RG |     MC-MED     | all conditions pooled |    6748.079    |   0.307   |      0.242      |     0.061     |
# | N2 20250417 COI |     MC-LAT     | all conditions pooled |    6086.179    |   0.352   |      0.311      |     0.053     |
# | N2 20250417 COI |     MC-MED     | all conditions pooled |    5818.160    |   0.311   |      0.258      |     0.058     |
# | N2 20250422 COI |     MC-LAT     | all conditions pooled |    6632.944    |   0.344   |      0.291      |     0.054     |
# | N2 20250422 COI |     MC-MED     | all conditions pooled |    3808.976    |   0.340   |      0.287      |     0.057     |
# | N2 20250509 COI |     MC-LAT     | all conditions pooled |    7365.118    |   0.313   |      0.251      |     0.061     |
# | N2 20250509 COI |     MC-MED     | all conditions pooled |    7086.714    |   0.291   |      0.240      |     0.064     |
# |    N1 MC AVG    |       MC       | all conditions pooled |    4876.287    |   0.189   |      0.140      |     0.058     |
# |  N2 MC-LAT AVG  |     MC-LAT     | all conditions pooled |    5550.025    |   0.347   |      0.290      |     0.059     |
# |  N2 MC-MED AVG  |     MC-MED     | all conditions pooled |    4523.969    |   0.321   |      0.276      |     0.055     |
# +-----------------+----------------+-----------------------+----------------+-----------+-----------------+---------------+