"""Convenience routines for forecasting and reconstructing emissions/latents."""

import ipdb
import numpy as np



def remove_empty_trials(
    data):

    data_res = []

    for i in range(len(data)):

        trial_data = data[i]

        if len(trial_data):
            data_res.append(trial_data)

    return data_res


def forecast_inference(
    latent_data,
    trial_lengths,
    emissions_params,
    dynamics_params):
    """
    Predict future neural data using the trained SSM and the current neural data's latent representation.
    latent_data: np.ndarray, shape (# total samples, #   latent states)
    trial_lengths: list of int with length = # trials
    """
    emissions_matrix = emissions_params.weights ## shape (n_latent_dims, n_obs_dims)
    emissions_bias   = emissions_params.bias    ## shape (n_obs_dims,)
    dynamics_matrix  = dynamics_params.weights  ## shape (n_latent_dims, n_latent_dims)
    dynamics_bias    = dynamics_params.bias     ## shape (n_latent_dims,)

    data_next = []
    trial_length_sum = 0

    for trial_length in trial_lengths:

        ## Access data of the current trial
        latent_data_ = latent_data[trial_length_sum:trial_length_sum + trial_length, :]
        trial_length_sum += trial_length

        ## Use the current latent state to predict the next latent state
        latent_data_next_ = latent_data_ @ dynamics_matrix.T + dynamics_bias # NOTE: Need the transpose here. Otherwise results are terrible

        ## Remove the last entry from latent_data_next_ and prepend the first entry of latent_data_
        ## This is because we cannot observe the previous state of the first state in the sequence 
        ##   and do not need to predict the next state of the last state in the sequence
        latent_data_next_ = np.concatenate([latent_data_[0, None], latent_data_next_[:-1, :]], axis=0)

        ## Use the emissions matrix to predict the neural data from the predicted latent state
        data_next_ = latent_data_next_ @ emissions_matrix.T + emissions_bias
        
        data_next.append(data_next_)

    ## Concatenate the predicted neural data of all trials
    data_next = np.concatenate(data_next, axis=0)

    assert data_next.shape[0] == latent_data.shape[0]
    return data_next


def forecast_inference_SLDS(
    latent_data,  ## continuous states
    discrete_states,
    trial_lengths,
    dynamics_params,
    emissions_params,
    inputs=None):
    """
    Predict future neural data using the trained SSM and the current neural data's latent representation.
    latent_data: np.ndarray, shape (# total samples, # latent states)
    trial_lengths: list of int with length = # trials
    """
    dynamics_matrices        = dynamics_params.As   ## shape (n_latent_dims, n_obs_dims)
    dynamics_biases          = dynamics_params.bs   ## shape (n_obs_dims,)
    dynamics_input_matrices  = dynamics_params.Vs   ## shape (n_latent_dims, n_input_dims)
    emissions_matrices       = emissions_params.Cs  ## shape (n_latent_dims, n_latent_dims)
    emissions_bias           = emissions_params.ds  ## shape (n_latent_dims,)
    emissions_input_matrices = emissions_params.Fs  ## shape (n_latent_dims, n_input_dims)

    data_next_all = []
    for i_trial, trial_length in enumerate(trial_lengths):

        data_next = []
        for t in range(trial_length):

            ## Obtain the latent data and discrete state of the current timestamp in the current trial
            latent_data_ = latent_data[i_trial][t, :].reshape(1, -1)
            input_data_  = inputs[i_trial][t + 1, :].reshape(1, -1) if inputs is not None else None

            if not len(discrete_states):
                latent_data_next_ = latent_data_ @ dynamics_matrices[0].T + dynamics_biases[0]

                if input_data_ is not None:
                    latent_data_next_ += input_data_ @ dynamics_input_matrices[0].T

            else:
                discrete_state_ = discrete_states[i_trial][t]

                ## Use the current latent data to predict the next latent data
                latent_data_next_ = latent_data_ @ dynamics_matrices[discrete_state_].T + dynamics_biases[discrete_state_]

                if input_data_ is not None:
                    latent_data_next_ += input_data_ @ dynamics_input_matrices[discrete_state_].T

            ## Use the predicted next latent data to predict the next emissions
            data_next_ = latent_data_next_ @ emissions_matrices[0].T + emissions_bias[0]

            if input_data_ is not None:
                data_next_ += input_data_ @ emissions_input_matrices[0].T

            data_next.append(data_next_.flatten())

        data_next_all.append(np.array(data_next))

    assert len(data_next_all) == len(latent_data)
    return data_next_all


def dynamics_inference_SLDS(
    latent_data,  ## continuous states
    discrete_states,
    trial_lengths,
    dynamics_params,
    inputs=None):
    """
    Predict future neural data using the trained SSM and the current neural data's latent representation.
    latent_data: np.ndarray, shape (# total samples, # latent states)
    trial_lengths: list of int with length = # trials
    """
    dynamics_matrices       = dynamics_params.As   ## shape (n_latent_dims, n_obs_dims)
    dynamics_biases         = dynamics_params.bs   ## shape (n_obs_dims,)
    dynamics_input_matrices = dynamics_params.Vs   ## shape (n_latent_dims, n_input_dims)

    latent_next_all = []

    for i_trial, trial_length in enumerate(trial_lengths):

        latent_next = []
        for t in range(trial_length):

            ## Obtain the latent data and discrete state of the current timestamp in the current trial
            latent_data_ = latent_data[i_trial][t, :].reshape(1, -1)
            input_data_  = inputs[i_trial][t + 1, :].reshape(1, -1) if inputs is not None else None

            if not len(discrete_states):
                latent_data_next_ = latent_data_ @ dynamics_matrices[0].T + dynamics_biases[0]

                if input_data_ is not None:
                    latent_data_next_ += input_data_ @ dynamics_input_matrices[0].T

            else:
                discrete_state_ = discrete_states[i_trial][t]

                ## Use the current latent data to predict the next latent data
                latent_data_next_ = latent_data_ @ dynamics_matrices[discrete_state_].T + dynamics_biases[discrete_state_]
                
                if input_data_ is not None:
                    latent_data_next_ += input_data_ @ dynamics_input_matrices[discrete_state_].T

            latent_next.append(latent_data_next_.flatten())

        latent_next_all.append(np.array(latent_next))

    assert len(latent_next_all) == len(latent_data)
    return latent_next_all


def emissions_inference_SLDS(
    latent_data,  ## continuous states
    emissions_params,
    inputs=None):
    """
    Predict future neural data using the trained SSM and the current neural data's latent representation.
    latent_data: np.ndarray, shape (# total samples, # latent states)
    trial_lengths: list of int with length = # trials
    """
    emissions_matrices       = emissions_params.Cs  ## shape (n_latent_dims, n_latent_dims)
    emissions_bias           = emissions_params.ds  ## shape (n_latent_dims,)
    emissions_input_matrices = emissions_params.Fs  ## shape (n_latent_dims, n_input_dims)

    inferred_emissions_all = []
    for latent_data_, input_ in zip(latent_data, inputs):
        
        ## Use the emissions parameters to predict the neural data from the latent states
        inferred_emissions = latent_data_ @ emissions_matrices[0].T + emissions_bias[0]

        trial_length = latent_data_.shape[0]

        if inputs is not None:
            inferred_emissions += (input_ @ emissions_input_matrices[0].T)[1:trial_length + 1, :]

        inferred_emissions_all.append(inferred_emissions)

    assert len(inferred_emissions_all) == len(latent_data)
    return inferred_emissions_all


def forecast_inference_baseline(
    data,
    trial_lengths,
    window_size):
    """
    Predict the future data using the average of the previous n timestamps. 
        n is specified by window_size.
    data: np.ndarray, shape (# total samples, # neurons)
    """
    data_next = []
    trial_length_sum = 0

    for trial_length in trial_lengths:

        ## Access data of the current trial
        data_ = data[trial_length_sum:trial_length_sum + trial_length, :]
        trial_length_sum += trial_length

        ## Use the data average of the previous n timestamps for the next timestamp
        ## Note that the last timestamp in the original data is not used
        data_next_ = np.zeros_like(data_).astype(float)

        for i in range(window_size, data_.shape[0]):
            data_next_[i, :] = np.mean(data_[i - window_size:i, :], axis=0)

        ## Fill the first n timestamps with the original data
        data_next_[:window_size, :] = data_[:window_size, :]
        
        data_next.append(data_next_)
    
    ## Concatenate the predicted neural data of all trials
    data_next = np.concatenate(data_next, axis=0)

    assert data_next.shape == data.shape
    return data_next
