"""Shared plotting style settings and color palettes for all SLDS figures."""

import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
import scripts.config as config


# print(mpl.rcParams.keys())  # Uncomment to inspect available rcParams
# Set a Nature-style default look for all figures generated in this repo
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],   # Helvetica on macOS
    "font.size": 7,                 # 5–7 pt allowed
    "axes.linewidth": 0.25,
    "xtick.major.width": 0.25,
    "xtick.major.size": 1,
    "ytick.major.width": 0.25,
    "ytick.major.size": 1,
    "xtick.labelsize": 5,          # tick labels
    "ytick.labelsize": 5,
    "patch.linewidth": 0.25,
    "pdf.fonttype": 42,             # embed TrueType, keeps text selectable
    "ps.fonttype": 42,
    "savefig.pad_inches": 0,

    # "errorbar.linewidth" : 0.5,   # stems
    # "errorbar.capthick" : 0.5,   # caps
    # "errorbar.capsize" : 1
})
mm = 1/25.4  # convenient conversion from millimeters to inches for figsize


# -----------------------------------------------------------------------------
# Read parameters from config (mirrors analysis scripts for consistency)
# -----------------------------------------------------------------------------
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



# -----------------------------------------------------------------------------
# Reusable transparency/size defaults for scatter/line elements
# -----------------------------------------------------------------------------
alpha_point = 0.8
size_point  = 100
alpha_line_thin  = 0.3
alpha_line       = 0.6
alpha_line_thick = 0.9
size_line_thin  = 0.5
size_line       = 1.2
size_line_thick = 1.8
label_fontsize = 18



# -----------------------------------------------------------------------------
# Color themes grouped by task and condition (ballistic/sustained/near/far)
# -----------------------------------------------------------------------------
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
theme_leaf_light   = '#DEF4BF'
theme_leaf_mid     = '#B5DA71'
theme_leaf_dark    = '#9ACC3F'
theme_yellow_light = '#FFF4BD'
theme_yellow_mid   = '#F7E68D'
theme_yellow_dark  = '#F0D75B'
theme_indigo_light = '#CCDBFD'
theme_indigo_mid   = '#89A8F4'
theme_indigo_dark  = '#5D86E6'
theme_purple_light = '#C9C7F2'
theme_purple_mid   = '#898CDA'
theme_purple_dark  = '#5E61CC'

corals  = [theme_coral_light,  theme_coral_mid,  theme_coral_dark]
greens  = [theme_green_light,  theme_green_mid,  theme_green_dark]
blues   = [theme_blue_light,   theme_blue_mid,   theme_blue_dark]
oranges = [theme_orange_light, theme_orange_mid, theme_orange_dark]
leaves  = [theme_leaf_light,   theme_leaf_mid,   theme_leaf_dark]
yellows = [theme_yellow_light, theme_yellow_mid, theme_yellow_dark]
indigos = [theme_indigo_light, theme_indigo_mid, theme_indigo_dark]
purples = [theme_purple_light, theme_purple_mid, theme_purple_dark]


color_palettes = {
    'CenterStart': {
        'fast': blues,
        'slow': oranges,
    },
    'CenterStartInterleave': {
        'fast': indigos,
        'slow': yellows,
    },
    'RadialGrid': {
        'near': greens,
        'far':  corals,
    },
}

# Discrete state color stacks (indexing into hidden state IDs per task)
discrete_state_colors = {
    'CenterStart': {
        'fast': [theme_blue_dark, theme_blue_light, theme_blue_mid],
        'slow': [theme_orange_dark, theme_orange_light, theme_orange_mid],
    },
    'CenterStartInterleave': {
        'fast': [theme_indigo_dark, theme_indigo_light, theme_indigo_mid],
        'slow': [theme_yellow_dark, theme_yellow_light, theme_yellow_mid],
    },
    'RadialGrid': {
        'near': [theme_green_dark, theme_green_light, theme_green_mid],
        'far':  [theme_coral_dark, theme_coral_light, theme_coral_mid],
    },
}

# Continuous colormaps used for heatmaps/gradients per condition
color_maps = {
    'CenterStart': {
        'fast': mcolors.LinearSegmentedColormap.from_list('CO fast', [blues[2], blues[0]]),
        'slow': mcolors.LinearSegmentedColormap.from_list('CO slow', [oranges[2], oranges[0]]),
    },
    'CenterStartInterleave': {
        'fast': mcolors.LinearSegmentedColormap.from_list('CSI fast', [indigos[2], indigos[0]]),
        'slow': mcolors.LinearSegmentedColormap.from_list('CSI slow', [yellows[2], yellows[0]]),
    },
    'RadialGrid': {
        'near': mcolors.LinearSegmentedColormap.from_list('RG near', [greens[2], greens[0]]),
        'far':  mcolors.LinearSegmentedColormap.from_list('RG far',  [corals[2], corals[0]]),
    },
    'gray': {
        'fast': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
        'slow': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
        'near': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
        'far': mcolors.LinearSegmentedColormap.from_list('gray_cmap', [sns.color_palette('Greys')[0], sns.color_palette('Greys')[3]]),
    },
    'gray_r': {
        'fast': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
        'slow': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
        'near': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
        'far': mcolors.LinearSegmentedColormap.from_list('gray_r_cmap', [sns.color_palette('Greys')[3], sns.color_palette('Greys')[0]]),
    }
}

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

session_trajectory_viewing_angles = {
    'sub-N1_ses-20190412_tf_CenterStart' : {
        'MC' : {
            'same_speed': {
                'view1' : {
                    'fast': [-44, -150, 90],
                    'slow': [-44, -150, 90]},
                'view2' : {
                    'fast': [-130, -150, 0],
                    'slow': [-130, -150, 0]},
                'view1_d2' : {
                    'fast': [-120, 10, 30],
                    'slow': [-120, 10, 30]},
                'view2_d2' : {
                    'fast': [-160, 125, 15],
                    'slow': [-160, 125, 15]},
            },
            'joint': {
                'view1' : {
                    'fast': [-44, -150, 90],
                    'slow': [-44, -150, 90]},
                'view2' : {
                    'fast': [-130, -150, 0],
                    'slow': [-130, -150, 0]},
                'view1_d2' : {
                    'fast': [-120, 10, 30],
                    'slow': [-120, 10, 30]},
                'view2_d2' : {
                    'fast': [-160, 125, 15],
                    'slow': [-160, 125, 15]},
            }
        }
    }, 
    'sub-N1_ses-20190517_tf_CenterStart' : {
        'MC' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [10, 30, 0],
                    'slow': [10, 30, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [10, 30, 0],
                    'slow': [10, 30, 0]},
                'view1' : {
                    'fast': [60, 30, 0],
                    'slow': [60, 30, 0]},
            }
        }
    }, 
    'sub-N1_ses-20190528_tf_CenterStart' : {
        'MC' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-15, 75, 0],
                    'slow': [-15, 75, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [-15, 75, 0],
                    'slow': [-15, 75, 0]},
                'view1' : {
                    'fast': [-50, 60, 0],
                    'slow': [-50, 60, 0]},
            }
        }
    }, 
    'sub-N2_ses-20240516_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-140, 40, 0],
                    'slow': [125, -140, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [125, -140, 0],
                    'slow': [125, -140, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-55, -30, 0],
                    'slow': [-30, 120, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [-30, 120, 0],
                    'slow': [-30, 120, 0]},
            }
        },
    },
    'sub-N2_ses-20240530_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-130, -130, 0],
                    'slow': [65, 30, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [65, 30, 0],
                    'slow': [65, 30, 0]},
                'view1' : {
                    'fast': [15, -70, 0],
                    'slow': [15, -70, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [50, 10, 0],
                    'slow': [-40, 60, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [-40, 60, 0],
                    'slow': [-40, 60, 0]},
                'view1' : {
                    'fast': [15, 30, 0],
                    'slow': [15, 30, 0]},
            }
        }
    },
    'sub-N2_ses-20240816_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [10, -60, 0],
                    'slow': [170, -150, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [170, -150, 0],
                    'slow': [170, -150, 0]},
                'view1' : {
                    'fast': [230, -130, 0],
                    'slow': [230, -130, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [30, -40, 0],
                    'slow': [40, -30, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [40, -30, 0],
                    'slow': [40, -30, 0]},
                'view1' : {
                    'fast': [45, -125, 0],
                    'slow': [45, -125, 0]},
            }
        }
    },
    'sub-N2_ses-20240820_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [165, -50, 0],
                    'slow': [140, -70, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [140, -70, 0],
                    'slow': [140, -70, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-45, 30, 0],
                    'slow': [40, -120, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [40, -120, 0],
                    'slow': [40, -120, 0]},
            }
        }
    },
    'sub-N2_ses-20241015_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [165, -50, 0],
                    'slow': [155, -165, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [155, -165, 0],
                    'slow': [155, -165, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [30, 15, 0],
                    'slow': [40, -120, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [40, -120, 0],
                    'slow': [40, -120, 0]},
            }
        }
    },
    'sub-N2_ses-20241022_tf_CenterStart' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-50, -170, 0],
                    'slow': [-130, -40, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [-130, -40, 0],
                    'slow': [-130, -40, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-35, -35, 0],
                    'slow': [-20, -35, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [-20, -35, 0],
                    'slow': [-20, -35, 0]},
            }
        }
    },
    'sub-N2_ses-20241105_tf_RadialGrid' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [30, -30, 0],
                    'far' : [-30, -60, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [-30, -60, 0],
                    'far' : [-30, -60, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [-40, 40, 0],
                    'far' : [30, -25, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [30, -25, 0],
                    'far' : [30, -25, 0]},
            }
        }
    },
    'sub-N2_ses-20241211_tf_RadialGrid' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [30, -25, 0],
                    'far' : [50, -20, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [50, -20, 0],
                    'far' : [50, -20, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [-40, 60, 0],
                    'far' : [-125, 30, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [-125, 30, 0],
                    'far' : [-125, 30, 0]},
            }
        }
    },
    'sub-N2_ses-20250408_tf_RadialGrid' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [45, -150, 0],
                    'far' : [40, 20, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [40, 20, 0],
                    'far' : [40, 20, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'near': [20, -110, 0],
                    'far' : [40, -130, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'near': [40, -130, 0],
                    'far' : [40, -130, 0]},
            }
        }
    },
    'sub-N2_ses-20250417_tf_CenterStartInterleave' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-55, -40, 0],
                    'slow': [55, -150, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [55, -150, 0],
                    'slow': [55, -150, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-60, -35, 0],
                    'slow': [50, -120, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [50, -120, 0],
                    'slow': [50, -120, 0]},
            }
        }
    },
    'sub-N2_ses-20250422_tf_CenterStartInterleave' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [65, 35, 0],
                    'slow': [50, 25, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [50, 25, 0],
                    'slow': [50, 25, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-60, -155, 0],
                    'slow': [45, -55, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [45, -55, 0],
                    'slow': [45, -55, 0]},
            }
        }
    },
    'sub-N2_ses-20250509_tf_CenterStartInterleave' : {
        'MC-LAT' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [-30, -35, 0],
                    'slow': [25, 55, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [25, 55, 0],
                    'slow': [25, 55, 0]},
                'view1' : {
                    'fast': [-35, 55, 0],
                    'slow': [-35, 55, 0]},
            }
        },
        'MC-MED' : {
            'same_speed': {
                'view1_d2' : {
                    'fast': [30, 65, 0],
                    'slow': [30, 65, 0]},
            },
            'joint': {
                'view1_d2' : {
                    'fast': [30, 65, 0],
                    'slow': [30, 65, 0]},
            }
        }
    },
}


session_target_radii = {
    'sub-N1_ses-20190412_CenterStart' : 0.1, 
    'sub-N1_ses-20190517_CenterStart' : 0.1,
    'sub-N1_ses-20190528_CenterStart' : 0.1,
    'sub-N1_ses-20190412_tf_CenterStart' : 0.1, 
    'sub-N1_ses-20190517_tf_CenterStart' : 0.1,
    'sub-N1_ses-20190528_tf_CenterStart' : 0.1,
    'sub-N2_ses-20240516_tf_CenterStart' : 0.075,
    'sub-N2_ses-20240530_tf_CenterStart' : 0.075,
    'sub-N2_ses-20240816_tf_CenterStart' : 0.075,
    'sub-N2_ses-20240820_tf_CenterStart' : 0.075,
    'sub-N2_ses-20241015_tf_CenterStart' : 0.075,
    'sub-N2_ses-20241022_tf_CenterStart' : 0.075,
    'sub-N2_ses-20241105_tf_RadialGrid' : 0.06,
    'sub-N2_ses-20241211_tf_RadialGrid' : 0.06,
    'sub-N2_ses-20250408_tf_RadialGrid' : 0.06,
    'sub-N2_ses-20250417_tf_CenterStartInterleave' : 0.075,
    'sub-N2_ses-20250422_tf_CenterStartInterleave' : 0.075,
    'sub-N2_ses-20250509_tf_CenterStartInterleave' : 0.075,
}
