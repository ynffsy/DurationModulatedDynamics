import ipdb

import numpy as np

import ssm
import autograd.numpy.random as npr



class SLDS:

    def __init__(
        self, 
        train_emissions,
        train_inputs,
        n_neurons, 
        n_input_neurons,
        data_format,
        random_state,
        n_continuous_states, 
        n_discrete_states,
        n_iters, 
        model_type,
        dynamics_class,
        emission_class,
        init_type,
        subspace_type,
        alpha):

        npr.seed(random_state)

        self.train_emissions = [np.array(emission) for emission in train_emissions]

        if train_inputs is not None:
            self.train_inputs = [np.array(input_) for input_ in train_inputs]
        else:
            self.train_inputs = None
        
        self.data_format         = data_format

        self.n_continuous_states = n_continuous_states
        self.n_discrete_states   = n_discrete_states
        self.n_iters             = n_iters

        self.model_type          = model_type
        self.dynamics_class      = dynamics_class
        self.init_type           = init_type
        self.subspace_type       = subspace_type
        self.alpha               = alpha
        
        self.emission_class      = emission_class
        self.emission_optimizer  = 'lbfgs'

        if self.subspace_type == 'single' or self.subspace_type is None:
            self.single_subspace = True
        elif self.subspace_type == 'multi':
            self.single_subspace = False
        else:
            ValueError(f"[SLDS.__init__()] Invalid subspace type '{self.subspace_type}' found")

        ## Use the default emission class if None
        if self.emission_class is None:
            self.emission_class = 'gaussian_orthog'

        ## Convert emissions to integers if the emission class is poisson
        if self.emission_class in ['poisson', 'poisson_orthog']:
            self.train_emissions = [np.array(emission).astype(int) for emission in self.train_emissions]

        if model_type == 'SLDS':
            self.model = ssm.SLDS(
                N=n_neurons, 
                K=n_discrete_states, 
                D=n_continuous_states, 
                M=n_input_neurons,
                emissions=self.emission_class)
            
        elif model_type == 'roSLDS':
            self.model = ssm.SLDS(
                N=n_neurons, 
                K=n_discrete_states, 
                D=n_continuous_states, 
                M=n_input_neurons,
                transitions="recurrent_only",
                dynamics=self.dynamics_class,
                emissions=self.emission_class,
                single_subspace=self.single_subspace)
            
        elif model_type == 'rSLDS':
            self.model = ssm.SLDS(
                N=n_neurons, 
                K=n_discrete_states, 
                D=n_continuous_states, 
                M=n_input_neurons,
                transitions="recurrent",
                dynamics=self.dynamics_class,
                emissions=self.emission_class,
                single_subspace=self.single_subspace)
            
        elif model_type == 'rSpLDS':
            self.model = ssm.SLDS(
                N=n_neurons, 
                K=n_discrete_states, 
                D=n_continuous_states, 
                M=n_input_neurons,
                transitions="recurrent",
                dynamics=self.dynamics_class,
                emissions=self.emission_class,
                emission_kwargs=dict(link='softplus'),
                single_subspace=self.single_subspace)
            
        elif model_type == 'rbfrSLDS':
            self.model = ssm.SLDS(
                N=n_neurons, 
                K=n_discrete_states, 
                D=n_continuous_states, 
                M=n_input_neurons,
                transitions="rbf_recurrent",
                dynamics=self.dynamics_class,
                emissions=self.emission_class,
                single_subspace=self.single_subspace)
            
        elif model_type == 'LDS':
            self.model = ssm.LDS(
                N=n_neurons, 
                D=n_continuous_states, 
                M=n_input_neurons,
                emissions=self.emission_class)
            
        elif model_type == 'pLDS':
            self.model = ssm.LDS(
                N=n_neurons, 
                D=n_continuous_states, 
                M=n_input_neurons,
                emissions=self.emission_class,
                emission_kwargs=dict(link='softplus'))

        else:
            ValueError(f"[SSM.__init__()] Invalid model type '{model_type}' found")
        
        ## TODO: Enable inputs and initializations


    def fit(
        self):
        """
        train_emissions: a list of firing rate arrays, each corresponding to a trial. 
            Each firing rate array has shape (# times, # neurons). 
            # times varies by trial; # neurons stays constant.
        """

        if self.init_type == 'ARHMM':
            self.train_elbos, self.train_posterior = self.model.fit(
                self.train_emissions, 
                inputs=self.train_inputs,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=self.n_iters, 
                alpha=self.alpha,
                emission_optimizer=self.emission_optimizer,
                initialize=True,
                num_init_iters=self.n_iters)
        
        elif self.init_type == 'emissions':
            self.train_elbos, self.train_posterior = self.model.fit(
                self.train_emissions, 
                inputs=self.train_inputs,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=self.n_iters, 
                alpha=self.alpha,
                emission_optimizer=self.emission_optimizer,
                initialize='emissions',
                num_init_iters=self.n_iters)
            
        elif self.init_type == 'transitions':
            self.train_elbos, self.train_posterior = self.model.fit(
                self.train_emissions, 
                inputs=self.train_inputs,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=self.n_iters, 
                alpha=self.alpha,
                emission_optimizer=self.emission_optimizer,
                initialize='transitions',
                num_init_iters=self.n_iters)
            
        elif self.init_type is None:
            self.train_elbos, self.train_posterior = self.model.fit(
                self.train_emissions, 
                inputs=self.train_inputs,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                num_iters=self.n_iters, 
                alpha=self.alpha,
                emission_optimizer=self.emission_optimizer,
                initialize=False,
                num_init_iters=self.n_iters)
        
        else:
            ValueError(f"[SSM.fit()] Invalid init type '{self.init_type}' found")

        ## Obtain the continuous latent states
        self.train_continuous_states = self.train_posterior.mean_continuous_states
        
        ## Obtain the continuous latent state covariances
        self.train_continuous_expectations = self.train_posterior.continuous_expectations

        self.train_continuous_state_covariances = []
        for i, exp in enumerate(self.train_continuous_expectations):
            self.train_continuous_state_covariances.append(exp[2])
    
        ## Obtain the discrete latent states
        self.train_discrete_states = [] 

        if self.model_type not in ['LDS', 'pLDS']:
            for i, train_continuous_states_ in enumerate(self.train_continuous_states):
                if self.train_inputs is None:
                    self.train_discrete_states.append(self.model.most_likely_states(train_continuous_states_, self.train_emissions[i]))
                else:
                    self.train_discrete_states.append(self.model.most_likely_states(train_continuous_states_, self.train_emissions[i], input=self.train_inputs[i]))
                # self.train_smoothing = self.model.smooth(self.train_continuous_states, self.train_emissions)


    def transform(
        self, 
        test_emissions=None,
        test_inputs=None):
        """
        Obtain the hidden states
        
        test_emissions == None -> use train_emissions (and train_inputs if applicable)
        test_emissions != None and test_inputs == None -> use test_emissions without inputs
        test_emissions != None and test_inputs != None -> use test_emissions with inputs
        """

        def reformat_data(data):

            if isinstance(data, list):
                return data
            elif isinstance(data, np.ndarray) and data.ndim == 3:
                return [np.array(data_) for data_ in data]
            elif isinstance(data, np.ndarray) and data.ndim == 2:
                return [data]
            else:
                ValueError(f"[SLDS.transform().reformat_data()] Invalid datas shape found")

        if test_emissions is None:
            emissions = self.train_emissions
            inputs = self.train_inputs
        else:
            if test_inputs is None:
                emissions = reformat_data(test_emissions)
                inputs = None
            else:
                emissions = reformat_data(test_emissions)
                inputs = reformat_data(test_inputs)

        ## Convert emissions to integers if the emission class is poisson
        if self.emission_class in ['poisson', 'poisson_orthog']:
            emissions = [np.array(emission).astype(int) for emission in emissions]

        self.test_elbos, self.test_posterior = self.model.approximate_posterior(
            emissions,
            inputs=inputs,
            method="laplace_em",
            variational_posterior="structured_meanfield",
            num_iters=self.n_iters, 
            alpha=self.alpha,
            emission_optimizer=self.emission_optimizer)
        
        ## Obtain the continuous latent states
        self.test_continuous_states = self.test_posterior.mean_continuous_states
        
        ## Obtain the continuous latent state covariances
        self.test_continuous_expectations = self.test_posterior.continuous_expectations

        self.test_continuous_state_covariances = []
        for i, exp in enumerate(self.test_continuous_expectations):
            self.test_continuous_state_covariances.append(exp[2])

        ## Obtain the discrete latent states
        self.test_discrete_states = []

        if self.model_type not in ['LDS', 'pLDS']:
            for i, test_continuous_states_ in enumerate(self.test_continuous_states):
                if inputs is None:
                    self.test_discrete_states.append(self.model.most_likely_states(test_continuous_states_, emissions[i]))
                else:
                    self.test_discrete_states.append(self.model.most_likely_states(test_continuous_states_, emissions[i], input=inputs[i]))


    def causal_transform(self, emissions, inputs=None):
        """
        y : (T,N) array
        u : (T,M) array or None
        Returns lists:
            x_filter_means[t]   ≈ E[x_t | y_{1:t}]
            z_probs[t]          ≈ p(z_t | y_{1:t})       (shape (K,))
        """

        n_trials = len(emissions)

        continuous_states_causal = []
        discrete_states_causal   = []

        for i_trial in range(n_trials):
            emissions_trial = emissions[i_trial]
            inputs_trial    = None if inputs is None else inputs[i_trial]

            T = emissions_trial.shape[0]

            continuous_states_causal_ = []
            discrete_states_causal_   = []

            for t in range(1, T + 1):
                y_pref = emissions_trial[:t]
                u_pref = None if inputs_trial is None else inputs_trial[:t]

                _, posterior = self.model.approximate_posterior(
                    [y_pref], 
                    inputs=[u_pref] if u_pref is not None else None,
                    method="laplace_em",
                    variational_posterior="structured_meanfield",
                    num_iters=self.n_iters,
                    alpha=self.alpha,
                    emission_optimizer=self.emission_optimizer)

                continuous_states_causal_.append(posterior.mean_continuous_states[0][-1])  # x_t
                discrete_states_causal_.append(np.argmax(posterior.mean_discrete_states[0][-1]))              # p(z_t)

            continuous_states_causal.append(np.array(continuous_states_causal_))
            discrete_states_causal.append(np.array(discrete_states_causal_))

        self.causal_continuous_states = continuous_states_causal
        self.causal_discrete_states   = discrete_states_causal
