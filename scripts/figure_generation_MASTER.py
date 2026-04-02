import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vis_paper
import statistics_paper
import config_SfN2024 as config



if __name__ == '__main__':

    session_info = {
        'N1_CenterStart': {
            'subject': 'N1',
            'task': 'CenterStart',
            'task_short': 'CO',
            'session_data_names': [
                'sub-N1_ses-20190412_tf_CenterStart',
                'sub-N1_ses-20190517_tf_CenterStart',
                'sub-N1_ses-20190528_tf_CenterStart',
            ],
            'unit_filters': ['MC'],
            'unit_filters_short': ['MC'],
            'window_configs': ['gt_0.132_fct_0.0_s0.01_gaussian_0.1_10'],
            'trial_filters': ['fast', 'slow'],
            'visual_delay_times': [0.132],
            'peak_times': [0.186]},
        'N2_CenterStart': {
            'subject': 'N2',
            'task': 'CenterStart',
            'task_short': 'CO',
            'session_data_names': [
                'sub-N2_ses-20240516_tf_CenterStart',
                'sub-N2_ses-20240530_tf_CenterStart',
                'sub-N2_ses-20240816_tf_CenterStart',
                'sub-N2_ses-20240820_tf_CenterStart',
                'sub-N2_ses-20241015_tf_CenterStart',
                'sub-N2_ses-20241022_tf_CenterStart',
            ],
            'unit_filters': [
                'MC-LAT', 
                'MC-MED',
            ],
            'unit_filters_short': [
                'MCL', 
                'MCM',
            ],
            'window_configs': [
                'gt_0.275_fct_0.0_s0.01_gaussian_0.1_10', 
                'gt_0.248_fct_0.0_s0.01_gaussian_0.1_10',
            ],
            'trial_filters': ['fast', 'slow'],
            ## Outdated
            # 'visual_delay_times': [
            #     0.290, 
            #     0.275,
            # ],
            # 'peak_times': [
            #     0.347, 
            #     0.321,
            # ]},
            'visual_delay_times': [
                0.275, 
                0.248,
            ],
            'peak_times': [
                0.354, 
                0.315,
            ]},
        'N2_CenterStartInterleave': {
            'subject': 'N2',
            'task': 'CenterStartInterleave',
            'task_short': 'COI',
            'session_data_names': [
                'sub-N2_ses-20250417_tf_CenterStartInterleave',
                'sub-N2_ses-20250422_tf_CenterStartInterleave',
                'sub-N2_ses-20250509_tf_CenterStartInterleave'
            ],
            'unit_filters': [
                'MC-LAT',
                'MC-MED',
                # 'PPC-SPL',
                # 'PPC-IPL',
            ],
            'unit_filters_short': [
                'MCL',
                'MCM',
                # 'SPL',
                # 'SMG',
            ],
            'window_configs': [
                'gt_0.275_fct_0.0_s0.01_gaussian_0.1_10',
                'gt_0.248_fct_0.0_s0.01_gaussian_0.1_10',
            ],
            'trial_filters': ['fast', 'slow'],
            'visual_delay_times': [
                0.275,
                0.248,
            ],
            'peak_times': [
                0.354,
                0.315,
            ]},
        'N2_RadialGrid': {
            'subject': 'N2',
            'task': 'RadialGrid',
            'task_short': 'RG',
            'session_data_names': [
                'sub-N2_ses-20241105_tf_RadialGrid',
                'sub-N2_ses-20241211_tf_RadialGrid',
                'sub-N2_ses-20250408_tf_RadialGrid',
            ],
            'unit_filters': ['MC-LAT', 'MC-MED'],
            'unit_filters_short': ['MCL', 'MCM'],
            'window_configs': ['gt_0.275_fct_0.0_s0.01_gaussian_0.1_10', 'gt_0.248_fct_0.0_s0.01_gaussian_0.1_10'],
            'trial_filters': ['near', 'far'],
            'visual_delay_times': [
                0.275,
                0.248,
            ],
            'peak_times': [
                0.354,
                0.315,
            ]},
    }



    #### Behavioral figures ####
    # time_step = 0.01
    # data_format = None

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filter        = session_info[session_key]['unit_filters'][0]
    #     window_config      = session_info[session_key]['window_configs'][0]
    #     trial_filters      = session_info[session_key]['trial_filters']

        # vis_paper.plot_behavioral_distances_to_target(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters)
        
        # vis_paper.plot_behavioral_target_acquisition_times(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters)

        # vis_paper.plot_behavioral_target_acquisition_times_pooled(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters)

        # vis_paper.plot_behavioral_cursor_velocity(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters,
        #     aggregate='trials')

        # vis_paper.plot_behavioral_cursor_velocity(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters,
        #     aggregate='both')

        # vis_paper.plot_behavioral_distances_to_target(
        #     session_data_names,
        #     unit_filter,
        #     window_config,
        #     data_format,
        #     trial_filters,
        #     aggregate='both')



    #### Single neuron examples ####
    # window_config = 'gt_0.0_fct_0.0_s0.01_gaussian_0.1_10'
    # time_step = 0.01
    # data_format = 'truncate_end'

    # example_neuron_indices = {
    #     'sub-N1_ses-20190412_tf_CenterStart': [37],
    #     'sub-N1_ses-20190517_tf_CenterStart': [39],
    #     'sub-N1_ses-20190528_tf_CenterStart': [15],
    # }

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     trial_filters      = session_info[session_key]['trial_filters']
    #     visual_delay_times = session_info[session_key]['visual_delay_times']
    #     peak_times         = session_info[session_key]['peak_times']

    #     for unit_filter, visual_delay_time, peak_time in zip(
    #             unit_filters, visual_delay_times, peak_times):

    #         for session_data_name in session_data_names:
    #             if session_data_name in example_neuron_indices:
    #                 vis_paper.plot_neurons_by_target_single_session(
    #                     session_data_name,
    #                     unit_filter,
    #                     window_config,
    #                     time_step,
    #                     data_format,
    #                     trial_filters,
    #                     neuron_indices=example_neuron_indices[session_data_name],
    #                     truncate_percentile=90,
    #                     visual_delay_time=visual_delay_time,
    #                     peak_time=peak_time)
    


    ### Single and population neuron figures ####
    # window_config = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'
    # time_step = 0.001
    # data_format = 'truncate_end'

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     trial_filters      = session_info[session_key]['trial_filters']
    #     visual_delay_times = session_info[session_key]['visual_delay_times']
    #     peak_times         = session_info[session_key]['peak_times']

        # for unit_filter, visual_delay_time, peak_time in zip(
        #         unit_filters, visual_delay_times, peak_times):

            # for session_data_name in session_data_names:
            #     vis_paper.plot_neural_speed_single_session(
            #         session_data_name,
            #         unit_filter,
            #         window_config,
            #         time_step,
            #         trial_filters,
            #         truncate_percentile=90,
            #         pre_start_time_buffer=0.2,
            #         post_reach_time_buffer=0.5,
            #         time_marker=visual_delay_time,
            #         save_stats=True,
            #         per_target_stats=True,
            #         small_format=False,
            #         peak_onset_time_nstd=1.5)

            # vis_paper.plot_time_crossnobis_RDM_matrix(
            #     session_data_names,
            #     unit_filter,
            #     window_config,
            #     time_step,
            #     data_format,
            #     trial_filters,
            #     truncate_percentile=90,
            #     pre_start_time_buffer=0.2,
            #     post_reach_time_buffer=0.5,
            #     visual_delay_time=visual_delay_time,
            #     peak_time=peak_time,
            #     supplement_format=True,
            #     per_target=True,
            #     show_phase_bars=True)

            # vis_paper.plot_neural_speed(
            #     session_data_names,
            #     unit_filter,
            #     window_config,
            #     time_step,
            #     trial_filters,
            #     truncate_percentile=90,
            #     pre_start_time_buffer=0.2,
            #     post_reach_time_buffer=0.5,
            #     superdiagonal_order=1,
            #     time_marker=visual_delay_time,
            #     visual_delay_time=visual_delay_time,
            #     peak_time=peak_time,
            #     supplement_format=True,
            #     per_target=True)

            # vis_paper.plot_cross_condition_crossnobis_RDM_matrix(
            #     session_data_names,
            #     unit_filter,
            #     window_config,
            #     time_step,
            #     data_format,
            #     trial_filters,
            #     truncate_percentile=90,
            #     pre_start_time_buffer=0.2,
            #     post_reach_time_buffer=0.5,
            #     visual_delay_time=visual_delay_time,
            #     peak_time=peak_time,
            #     per_target=True,
            #     supplement_format=True)

            # statistics_paper.cross_condition_significance_crossnobis(
            #     session_data_names,
            #     unit_filter,
            #     window_config,
            #     time_step,
            #     data_format,
            #     trial_filters,
            #     truncate_percentile=90,
            #     pre_start_time_buffer=0.2,
            #     post_reach_time_buffer=0.5,
            #     visual_delay_time=visual_delay_time,
            #     peak_time=peak_time,
            #     significance_method='wilcoxon',  # 'wilcoxon', 'ttest', 'permutation'
            #     per_target=True,
            #     show_individual_targets=True,
            #     plot_heatmap=True,
            #     downsample_factor=1,
            #     supplement_format=True)

        # statistics_paper.neural_speed_statistics(
        #     session_data_names,
        #     unit_filters,
        #     window_config,
        #     time_step,
        #     trial_filters,
        #     truncate_percentile=90,
        #     pre_start_time_buffer=0.2,
        #     post_reach_time_buffer=0.5,
        #     superdiagonal_order=1,
        #     per_target=True)


    # window_config = 'gt_0.0_fct_0.0_s0.01_gaussian_0.1_10'
    # time_step = 0.01
    # data_format = 'truncate_end'
    # reaction_time = 0.0

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     trial_filters      = session_info[session_key]['trial_filters']
    #     visual_delay_times = session_info[session_key]['visual_delay_times']
    #     peak_times         = session_info[session_key]['peak_times']

    #     for unit_filter, visual_delay_time, peak_time in zip(
    #             unit_filters, visual_delay_times, peak_times):
            
    #         n_sessions = len(session_data_names)
    #         reaction_times_ = [reaction_time] * n_sessions
    #         visual_delay_times_ = [visual_delay_time] * n_sessions
    #         peak_times_ = [peak_time] * n_sessions

    #         vis_paper.plot_percent_neuron_discrepancy(
    #             session_data_names,
    #             unit_filter,
    #             time_step,
    #             window_config,
    #             data_format,
    #             trial_filters,
    #             reaction_times_,
    #             visual_delay_times_,
    #             peak_times_,
    #             truncate_percentile=90)

    # vis_paper.plot_percent_neuron_discrepancy_all_sessions(
    #     session_info,
    #     time_step,
    #     window_config,
    #     data_format,
    #     reaction_time=reaction_time,
    #     truncate_percentile=90)



    #### LDS figures ####

    # input_unit_filter = 'PPC-SPL'
    input_unit_filter = None
    time_step = 0.01
    data_format = None
    train_test_option = 'joint'
    # train_test_option = 'same_speed'
    random_state = 42
    n_continuous_states = 3
    # n_continuous_states = 20
    n_discrete_states = 2
    n_iters = 25
    model_type = 'rSLDS'
    dynamics_class = 'diagonal_gaussian'
    emission_class = 'gaussian'
    init_type = 'ARHMM'
    # init_type = 'emissions'
    subspace_type = None
    alpha = 0.5

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     window_configs     = session_info[session_key]['window_configs']
    #     trial_filters      = session_info[session_key]['trial_filters']
    #     visual_delay_times = session_info[session_key]['visual_delay_times']
    #     peak_times         = session_info[session_key]['peak_times']

        # for session_data_name in session_data_names:
        #     for unit_filter, window_config, visual_delay_time in zip(unit_filters, window_configs, visual_delay_times):

                ## F3 abcd
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     3,
                #     1, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=True,
                #     show_average_trajectories=True,
                #     color_by_time_gradient=False,
                #     color_by_discrete_state=False,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=False,
                #     normalize_flow_field=False,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view2', 
                #     visual_delay_time=visual_delay_time,
                #     show_preview=True)
                
                # F3 efgh
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     3,
                #     1, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=True,
                #     show_average_trajectories=True,
                #     color_by_time_gradient=True,
                #     color_by_discrete_state=False,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=False,
                #     normalize_flow_field=False,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view2',
                #     visual_delay_time=visual_delay_time)

                # F4 ab
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     3,
                #     2, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=True,
                #     show_average_trajectories=False,
                #     color_by_time_gradient=False,
                #     color_by_discrete_state=True,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=False,
                #     normalize_flow_field=False,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view1_d2',
                #     visual_delay_time=visual_delay_time)
                
                # F4 cd
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     3,
                #     2, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=False,
                #     show_average_trajectories=True,
                #     color_by_time_gradient=False,
                #     color_by_discrete_state=True,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=True,
                #     normalize_flow_field=True,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view2_d2',
                #     visual_delay_time=visual_delay_time,
                #     show_preview=True)

                # for trial_filter in trial_filters:

                #     ## F4 hi
                #     vis_paper.plot_cursor_trajectories(
                #         session_data_name,
                #         unit_filter,
                #         input_unit_filter,
                #         window_config,
                #         data_format,
                #         trial_filter,
                #         train_test_option,
                #         random_state,
                #         n_continuous_states,
                #         n_discrete_states,
                #         n_iters,
                #         model_type,
                #         dynamics_class,
                #         emission_class,
                #         init_type,
                #         subspace_type,
                #         alpha,
                #         show_target_positions=True,
                #         discrete_state_overlay=True)


                # Supplement figures - time gradient
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     n_continuous_states,
                #     n_discrete_states, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=False,
                #     show_average_trajectories=True,
                #     color_by_time_gradient=True,
                #     color_by_discrete_state=False,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=False,
                #     normalize_flow_field=False,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view1',
                #     visual_delay_time=visual_delay_time,
                #     supplement_format=True,
                #     show_preview=False)
                
                ## Supplement figures - discrete state
                # vis_paper.plot_3D_dynamical_latent_trajectories_integrated(
                #     session_data_name,
                #     unit_filter,
                #     input_unit_filter,
                #     window_config,
                #     time_step,
                #     data_format,
                #     trial_filters,
                #     train_test_option,
                #     random_state,
                #     n_continuous_states,
                #     n_discrete_states, 
                #     n_iters,
                #     model_type,
                #     dynamics_class,
                #     emission_class,
                #     init_type,
                #     subspace_type,
                #     alpha,
                #     show_individual_trajectories=False,
                #     show_average_trajectories=True,
                #     color_by_time_gradient=False,
                #     color_by_discrete_state=True,
                #     time_index_marker=None,
                #     show_turning_points=False,
                #     show_flow_field=True,
                #     normalize_flow_field=True,
                #     show_flow_field_boundary=False,
                #     show_custom_axes=True,
                #     view_name='view1_d2',
                #     visual_delay_time=visual_delay_time,
                #     supplement_format=True,
                #     show_preview=True)



        # for unit_filter, window_config, visual_delay_time, peak_time in zip(unit_filters, window_configs, visual_delay_times, peak_times):  

        #     vis_paper.plot_discrete_states_over_time(
        #         session_data_names,
        #         unit_filter,
        #         input_unit_filter,
        #         window_config,
        #         time_step,
        #         data_format,
        #         trial_filters,
        #         train_test_option,
        #         random_state,
        #         n_continuous_states,
        #         n_discrete_states, 
        #         n_iters,
        #         model_type,
        #         dynamics_class,
        #         emission_class,
        #         init_type,
        #         subspace_type,
        #         alpha,
        #         visual_delay_time=visual_delay_time,
        #     )


    #### Crossnobis robustness check ####

    # window_config_robustness = 'gt_-0.2_fct_0.5_s0.001_gaussian_0.03_10'
    # time_step_robustness = 0.001
    # data_format_robustness = 'truncate_end'
    # reaction_time = 0.0

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     trial_filters      = session_info[session_key]['trial_filters']
    #     visual_delay_times = session_info[session_key]['visual_delay_times']
    #     peak_times         = session_info[session_key]['peak_times']

    #     for unit_filter in unit_filters:
    #         n_sessions = len(session_data_names)
    #         statistics_paper.crossnobis_robustness_check(
    #             session_data_names,
    #             unit_filter,
    #             window_config_robustness,
    #             time_step_robustness,
    #             data_format_robustness,
    #             trial_filters,
    #             reaction_time,
    #             [visual_delay_times[0]] * n_sessions,
    #             [peak_times[0]] * n_sessions,
    #             truncate_percentile=90,
    #             per_target=True,
    #             recompute=False,
    #             recompute_reduced=False)


    #### Steady-phase target decoding ####

    # Subject name mapping for x-axis labels
    subject_display = {'N1': 'JJ', 'N2': 'RD'}

    all_decoding_results = []

    for session_key in session_info.keys():
        session_data_names = session_info[session_key]['session_data_names']
        unit_filters       = session_info[session_key]['unit_filters']
        unit_filters_short = session_info[session_key]['unit_filters_short']
        window_configs     = session_info[session_key]['window_configs']
        trial_filters      = session_info[session_key]['trial_filters']
        subject_short      = subject_display[session_info[session_key]['subject']]
        task_short         = session_info[session_key]['task_short']

        for unit_filter, unit_short, window_config in zip(
                unit_filters, unit_filters_short, window_configs):

            # Decoding (returns per-trial correctness + angular errors)
            trial_correct, angular_errors, results_df = \
                statistics_paper.steady_phase_target_decoding(
                    session_data_names,
                    unit_filter,
                    window_config,
                    trial_filters[1],  # slow / far condition
                    steady_window_ms=300,
                    n_cv_folds=5)

            label = f'{subject_short} {unit_short} {task_short}'
            chance = 1.0 / results_df['n_targets'].iloc[0] if len(results_df) > 0 else 0.125
            chance_angular_error = results_df['chance_angular_error'].iloc[0] if len(results_df) > 0 else 90.0
            all_decoding_results.append({
                'label': label,
                'trial_correct': trial_correct,
                'angular_errors': angular_errors,
                'chance': chance,
                'chance_angular_error': chance_angular_error,
            })

            # PCA trajectories and progress-to-target — session by session
            for session_data_name in session_data_names:
                statistics_paper.steady_phase_pca_trajectories(
                    session_data_name,
                    unit_filter,
                    window_config,
                    trial_filters[1],
                    steady_window_ms=300)

                # statistics_paper.steady_phase_progress_to_target(
                #     session_data_name,
                #     unit_filter,
                #     window_config,
                #     trial_filters[1],
                #     steady_window_ms=300)

    # Summary bar plot across all subjects/tasks
    statistics_paper.plot_steady_phase_decoding_summary(
        all_decoding_results,
        steady_window_ms=300)


    #### Skew-symmetric ratio of dynamics matrix A ####

    ## rSLDS (K=2) — all subjects/tasks on same plot
    # vis_paper.plot_skew_symmetric_ratio(
    #     session_info,
    #     input_unit_filter,
    #     data_format,
    #     'joint',
    #     random_state,
    #     n_continuous_states,
    #     2,  # ns_discrete_states: rSLDS (>=2) vs LDS (1)
    #     n_iters,
    #     'rSLDS',
    #     dynamics_class,
    #     emission_class,
    #     init_type,
    #     subspace_type,
    #     alpha,
    #     plot_eigenspectra=False,
    #     plot_skew_ratio=True,
    #     plot_angular_velocity=True,
    #     plot_rotational_fraction=False,
    #     combine_trial_filters=True)


    #### LDS vs rSLDS goodness of fit ####

    # for session_key in session_info.keys():
    #     session_data_names = session_info[session_key]['session_data_names']
    #     unit_filters       = session_info[session_key]['unit_filters']
    #     window_configs     = session_info[session_key]['window_configs']
    #     trial_filters      = session_info[session_key]['trial_filters']

    #     for unit_filter, window_config in zip(unit_filters, window_configs):
    #         statistics_paper.lds_vs_rslds_goodness_of_fit(
    #             session_data_names,
    #             unit_filter,
    #             input_unit_filter,
    #             window_config,
    #             data_format,
    #             trial_filters,
    #             train_test_option,
    #             random_state,
    #             n_continuous_states,
    #             [1, 2],  # ns_discrete_states: LDS (1) vs rSLDS (2)
    #             n_iters,
    #             dynamics_class,
    #             emission_class,
    #             init_type,
    #             subspace_type,
    #             alpha,
    #             inference_types=['forecast', 'emissions'])

    pass
