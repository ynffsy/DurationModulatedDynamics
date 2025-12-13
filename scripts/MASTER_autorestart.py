import os
import itertools
import subprocess

import scripts.config as config
import utils.utils_processing as utils_processing



## Read parameters from config
overwrite_results  = config.overwrite_results
data_dir           = config.data_dir
results_dir        = config.results_dir
session_data_names = config.session_data_names

unit_filters       = config.unit_filters
input_unit_filters = config.input_unit_filters
window_config      = config.window_config
time_offset        = config.time_offset
data_formats       = config.data_formats
trial_filters      = config.trial_filters

random_states      = config.random_states
n_folds            = config.n_folds
ns_states          = config.ns_states
ns_discrete_states = config.ns_discrete_states
ns_iters           = config.ns_iters

model_types        = config.model_types
dynamics_classes   = config.dynamics_classes
emission_classes   = config.emission_classes
init_types         = config.init_types
subspace_types     = config.subspace_types
alphas             = config.alphas 



def check_completion_joint(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha):

    data_loader = utils_processing.DataLoaderDuo(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filters)
    
    model_results_dir_self, model_results_dir_ctpt = data_loader.get_model_result_dirs(
        time_offset=time_offset,
        data_format=data_format,
        train_test='joint',
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=False)
    
    for random_state in random_states:
        for i_fold in range(n_folds):
            for n_continuous_states in ns_states:
                for n_discrete_states in ns_discrete_states:
                    for n_iters in ns_iters:

                        model_save_name = '_'.join(map(str, [x for x in [
                            'r' + str(random_state),
                            'f' + str(i_fold),
                            's' + str(n_continuous_states),
                            'd' + str(n_discrete_states),
                            'i' + str(n_iters)]]))
                        
                        model_save_prefix_self = os.path.join(model_results_dir_self, model_save_name)
                        model_save_prefix_ctpt = os.path.join(model_results_dir_ctpt, model_save_name)
                        res_save_path_self     = model_save_prefix_self + '.pkl'
                        res_save_path_ctpt     = model_save_prefix_ctpt + '.pkl'

                        if not (os.path.isfile(res_save_path_self) and os.path.isfile(res_save_path_ctpt)):
                            print('s' + str(n_continuous_states) + 'i' + str(n_iters) + ' model does not exist. Starting rerun ...')
                            return False
                        
    return True



def check_completion_same_speed(
    session_data_name,
    unit_filter,
    input_unit_filter,
    data_format,
    trial_filter,
    model_type,
    dynamics_class,
    emission_class,
    init_type,
    subspace_type,
    alpha):

    data_loader = utils_processing.DataLoader(
        data_dir,
        results_dir,
        session_data_name,
        unit_filter,
        input_unit_filter,
        window_config,
        trial_filter)
    
    model_results_dir = data_loader.get_model_result_dir(
        time_offset=time_offset,
        data_format=data_format,
        train_test='same_speed',
        model_type=model_type,
        dynamics_class=dynamics_class,
        emission_class=emission_class,
        init_type=init_type,
        subspace_type=subspace_type,
        alpha=alpha,
        check_existence=False)
    
    for random_state in random_states:
        for i_fold in range(n_folds):
            for n_continuous_states in ns_states:
                for n_discrete_states in ns_discrete_states:
                    for n_iters in ns_iters:

                        model_save_name = '_'.join(map(str, [x for x in [
                            'r' + str(random_state),
                            'f' + str(i_fold),
                            's' + str(n_continuous_states),
                            'd' + str(n_discrete_states),
                            'i' + str(n_iters)]]))
                        
                        model_save_prefix = os.path.join(model_results_dir, model_save_name)
                        res_save_path     = model_save_prefix + '.pkl'

                        if not os.path.isfile(res_save_path):
                            print('s' + str(n_continuous_states) + 'i' + str(n_iters) + ' model does not exist. Starting rerun ...')
                            return False
                        
    return True



if __name__ == '__main__':

    # for (
    #     session_data_name, 
    #     unit_filter, 
    #     input_unit_filter, 
    #     data_format,
    #     trial_filter, 
    #     model_type, 
    #     dynamics_class, 
    #     emission_class,
    #     init_type, 
    #     subspace_type,
    #     alpha) in itertools.product(
    #         session_data_names, 
    #         unit_filters, 
    #         input_unit_filters, 
    #         data_formats,
    #         trial_filters, 
    #         model_types, 
    #         dynamics_classes, 
    #         emission_classes,
    #         init_types, 
    #         subspace_types,
    #         alphas):

    #     while not check_completion_same_speed(
    #         session_data_name,
    #         unit_filter,
    #         input_unit_filter,
    #         data_format,
    #         trial_filter,
    #         model_type,
    #         dynamics_class,
    #         emission_class,
    #         init_type,
    #         subspace_type,
    #         alpha):

    #         subprocess.run(['python', 'same_speed_SLDS.py'])

    
    for (
        session_data_name, 
        unit_filter, 
        input_unit_filter, 
        data_format,
        model_type, 
        dynamics_class, 
        emission_class,
        init_type, 
        subspace_type,
        alpha) in itertools.product(
            session_data_names, 
            unit_filters, 
            input_unit_filters, 
            data_formats,
            model_types, 
            dynamics_classes, 
            emission_classes,
            init_types, 
            subspace_types,
            alphas):

        while not check_completion_joint(
            session_data_name,
            unit_filter,
            input_unit_filter,
            data_format,
            model_type,
            dynamics_class,
            emission_class,
            init_type,
            subspace_type,
            alpha):

            subprocess.run(['python', 'joint_SLDS.py'])
