import ipdb
import numpy as np
import multiprocessing as mp



def dsup_ratio(
    x_obs, 
    x_latent, 
    x_latent_cov, 
    trial_lengths,
    model_params):
    """
    DSUP ratio is the Dynamical State Update Process Ratio from https://www.nature.com/articles/ncomms8759
    Correspondence between variables used here and in the paper:
        x_obs        = y (Each time step is y_k)
        x_latent     = \hat{s} (Each time step is \hat{s}_k. The s_k is the theoretical true latent state)
        x_latent_cov = \Sigma (Covariance matrix of the latent state. Each time step is \Sigma_k)

        dynamics_mat = M (This is the state transition matrix in general. It should be square. Its dimension should be the same as the number of states)
        dynamics_cov = N (This is the covariance matrix of the state transition noise)
        obs_mat      = P 
        obs_cov      = R (This is the covariance matrix of the observation noise)
    """

    obs_mat      = model_params.emissions.weights.T
    obs_cov      = model_params.emissions.cov
    dynamics_mat = model_params.dynamics.weights.T
    dynamics_cov = model_params.dynamics.cov
    obs_cov_inv  = np.linalg.inv(obs_cov)

    dsupr_all = []
    trial_length_sum = 0

    for trial_length in trial_lengths:
        x_obs_        = x_obs[trial_length_sum:trial_length_sum + trial_length, :]
        x_latent_     = x_latent[trial_length_sum:trial_length_sum + trial_length, :]
        x_latent_cov_ = x_latent_cov[trial_length_sum:trial_length_sum + trial_length, :, :]
        trial_length_sum += trial_length

        x_obs_        = x_obs_[1:, :]
        x_latent_     = x_latent_[:-1, :]
        x_latent_cov_ = x_latent_cov_[:-1, :, :]

        ip_diff = x_obs_ - x_latent_ @ dynamics_mat @ obs_mat

        ## Compute the dynamical state update process contribution
        dsup = np.linalg.norm(x_latent_ @ (dynamics_mat - np.eye(dynamics_mat.shape[0])), axis=1)

        ## Compute the innovation process contribution time point by time point
        ip = []
        n_times = x_obs_.shape[0]
        for k in range(n_times):

            ## Compute the Kalman gain
            M_Sigma_M_T = dynamics_mat @ x_latent_cov_[k, :, :] @ dynamics_mat.T
            K_k = np.linalg.inv(np.eye(dynamics_mat.shape[0]) + (M_Sigma_M_T + dynamics_cov) @ obs_mat @ obs_cov_inv @ obs_mat.T) @ (M_Sigma_M_T + dynamics_cov) @ obs_mat @ obs_cov_inv
            
            ip_k = ip_diff[k, :] @ K_k.T
            ip.append(ip_k)

        ip = np.linalg.norm(np.array(ip), axis=1)
        
        ## Compute the mean DSUP ratio for the current trial
        dsupr = np.mean(dsup / (dsup + ip))

        dsupr_all.append(dsupr)

    return np.mean(dsupr_all)


def compute_dsup_ratio_for_trial(args):
    x_obs, x_latent, x_latent_cov, trial_start, trial_end, obs_mat, obs_cov_inv, dynamics_mat, dynamics_cov = args
    
    x_obs_        = x_obs[trial_start:trial_end, :]
    x_latent_     = x_latent[trial_start:trial_end, :]
    x_latent_cov_ = x_latent_cov[trial_start:trial_end, :, :]

    x_obs_        = x_obs_[1:, :]
    x_latent_     = x_latent_[:-1, :]
    x_latent_cov_ = x_latent_cov_[:-1, :, :]

    ip_diff = x_obs_ - x_latent_ @ dynamics_mat @ obs_mat

    # Compute the dynamical state update process contribution
    dsup = np.linalg.norm(x_latent_ @ (dynamics_mat - np.eye(dynamics_mat.shape[0])), axis=1)

    # Compute the innovation process contribution time point by time point
    ip = []
    n_times = x_obs_.shape[0]

    for k in range(n_times):
        M_Sigma_M_T = dynamics_mat @ x_latent_cov_[k, :, :] @ dynamics_mat.T
        K_k = np.linalg.inv(np.eye(dynamics_mat.shape[0]) + (M_Sigma_M_T + dynamics_cov) @ obs_mat @ obs_cov_inv @ obs_mat.T) @ (M_Sigma_M_T + dynamics_cov) @ obs_mat @ obs_cov_inv
        
        ip_k = ip_diff[k, :] @ K_k.T
        ip.append(ip_k)

    ip = np.linalg.norm(np.array(ip), axis=1)
    
    # Compute the mean DSUP ratio for the current trial
    dsupr = np.mean(dsup / (dsup + ip))
    return dsupr


def dsup_ratio_parallel(x_obs, x_latent, x_latent_cov, trial_lengths, model_params, num_processes=None):

    obs_mat      = np.array(model_params.emissions.weights).T
    obs_cov      = np.array(model_params.emissions.cov)
    dynamics_mat = np.array(model_params.dynamics.weights).T
    dynamics_cov = np.array(model_params.dynamics.cov)
    obs_cov_inv  = np.linalg.inv(obs_cov)

    trial_start_end_indices = []
    trial_length_sum = 0

    for trial_length in trial_lengths:
        trial_start       = trial_length_sum
        trial_end         = trial_length_sum + trial_length
        trial_length_sum += trial_length
        trial_start_end_indices.append((trial_start, trial_end))
    
    with mp.Pool(processes=num_processes) as pool:
        args = [(x_obs, x_latent, x_latent_cov, start, end, obs_mat, obs_cov_inv, dynamics_mat, dynamics_cov) for start, end in trial_start_end_indices]
        dsupr_all = pool.map(compute_dsup_ratio_for_trial, args)
        
    return np.mean(dsupr_all)


def dsup_ratio_SLDS(
    x_obs, 
    x_latent, 
    x_latent_cov, 
    obs_mat,
    obs_bias,
    obs_cov,
    dynamics_mats,
    dynamics_biases,
    dynamics_covs,
    discrete_states):
    """
    DSUP ratio is the Dynamical State Update Process Ratio from https://www.nature.com/articles/ncomms8759
    Correspondence between variables used here and in the paper:
        x_obs        = y (Each time step is y_k)
        x_latent     = \hat{s} (Each time step is \hat{s}_k. The s_k is the theoretical true latent state)
        x_latent_cov = \Sigma (Covariance matrix of the latent state. Each time step is \Sigma_k)

        dynamics_mat = M (This is the state transition matrix in general. It should be square. Its dimension should be the same as the number of states)
        dynamics_cov = N (This is the covariance matrix of the state transition noise)
        obs_mat      = P 
        obs_cov      = R (This is the covariance matrix of the observation noise)
    """

    obs_cov_inv  = np.linalg.inv(obs_cov)
    dsupr_all = []
    dsupr_all_per_time = []


    ## For testing purposes
    # K_all = []
    # ip_all = []
    # dsup_all = []
    # M_Sigma_M_T_all = []
    # ip_diff_all = []
    # dynamics_mat_all = []
    # obs_mat_all = []


    for i, x_obs_ in enumerate(x_obs):
        x_latent_     = x_latent[i]
        x_latent_cov_ = x_latent_cov[i]

        x_obs_        = x_obs_[1:, :]
        x_latent_     = x_latent_[:-1, :]
        x_latent_cov_ = x_latent_cov_[:-1, :, :]

        ## Compute time point by time point
        dsup = []
        ip   = []
        n_times = x_obs_.shape[0]
        for k in range(n_times):

            if not len(discrete_states):
                dynamics_mat  = dynamics_mats[0]
                dynamics_bias = dynamics_biases[0]
                dynamics_cov  = dynamics_covs[0]
            else:
                discrete_state_ = discrete_states[i][k]
                dynamics_mat  = dynamics_mats[discrete_state_]
                dynamics_bias = dynamics_biases[discrete_state_]
                dynamics_cov  = dynamics_covs[discrete_state_]

            ## Compute the Kalman gain
            M_Sigma_M_T = dynamics_mat @ x_latent_cov_[k, :, :] @ dynamics_mat.T
            K_k = np.linalg.inv(np.eye(dynamics_mat.shape[0]) + (M_Sigma_M_T + dynamics_cov) @ obs_mat.T @ obs_cov_inv @ obs_mat) @ (M_Sigma_M_T + dynamics_cov) @ obs_mat.T @ obs_cov_inv
            
            ## Without bias term
            # ip_diff = x_obs_[k, :] - obs_mat @ dynamics_mat @ x_latent_[k]

            ## With bias term
            ip_diff = x_obs_[k, :] - (obs_mat @ (dynamics_mat @ x_latent_[k] + dynamics_bias) + obs_bias)

            ## Compute the innovation process contribution
            ip_k = K_k @ ip_diff
            ip_k_norm = np.linalg.norm(ip_k)
            ip.append(ip_k_norm)

            ## Compute the dynamical state update process contribution (without bias term)
            # dsup_k = (dynamics_mat - np.eye(dynamics_mat.shape[0])) @ x_latent_[k]

            ## Compute the dynamical state update process contribution (with bias term)
            dsup_k = (dynamics_mat - np.eye(dynamics_mat.shape[0])) @ x_latent_[k] + dynamics_bias

            dsup_k_norm = np.linalg.norm(dsup_k)
            dsup.append(dsup_k_norm)

            dsupr_all_per_time.append(dsup_k_norm / (dsup_k_norm + ip_k_norm))

            ## For testing purposes
            # K_all.append(K_k)
            # ip_all.append(ip_k)
            # dsup_all.append(dsup_k)
            # M_Sigma_M_T_all.append(M_Sigma_M_T)
            # ip_diff_all.append(ip_diff)
            # dynamics_mat_all.append(dynamics_mat)
            # obs_mat_all.append(obs_mat)

        ip = np.array(ip)
        dsup = np.array(dsup)
        
        ## Compute the mean DSUP ratio for the current trial
        dsupr = (dsup / (dsup + ip))

        dsupr_all += dsupr.tolist()

    # K_all = np.array(K_all)
    # ip_all = np.array(ip_all)
    # dsup_all = np.array(dsup_all)
    # M_Sigma_M_T_all = np.array(M_Sigma_M_T_all)
    # ip_diff_all = np.array(ip_diff_all)
    # dynamics_mat_all = np.array(dynamics_mat_all)
    # obs_mat_all = np.array(obs_mat_all)


    # ipdb.set_trace()

    return np.mean(dsupr_all), np.array(dsupr_all_per_time)
