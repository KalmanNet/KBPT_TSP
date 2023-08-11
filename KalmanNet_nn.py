"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

# from filing_paths import path_model
# import sys
# sys.path.insert(1, path_model)
# from model import getJacobian

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")




class KalmanNetNN_arch1(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learnable = True

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):


        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    ###################################
    ### Set if the NET is learnable ###
    ###################################
    def set_is_learnable(self, is_learnable):
        self.learnable = is_learnable
        for var in list(self.parameters()):
            var.requires_grad = is_learnable


    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)


        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(self.F, self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior

        self.inner_state_debug={}

        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)
        self.update_inner_state_debug('F1 dm1x', dm1x_reshape)
        self.update_inner_state_debug('F1 dm1x_norm', dm1x_norm)


        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)
        self.update_inner_state_debug('F1 dm1y', dm1y)
        self.update_inner_state_debug('F1 dm1y_norm', dm1y_norm)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)
        self.update_inner_state_debug('KGainNet_in', KGainNet_in)


        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        self.H = torch.tensor([y[0].item(),1]).reshape(1,-1)
        y = y[1:2].clone()


        # Compute Priors
        self.step_prior()


        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y
        self.dy = dy

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);
        self.update_inner_state_debug('La1_out', La1_out)


        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))
        self.update_inner_state_debug('GRU_out', GRU_out_reshape)


        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)
        self.update_inner_state_debug('La2_out', La2_out)


        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        self.update_inner_state_debug('Gain', L3_out)

        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

    def update_inner_state_debug(self,name,variable):
        variable = variable.detach().squeeze().reshape(-1,1)
        if variable.shape[0]>1:
            for i in range(variable.shape[0]):
                self.inner_state_debug[f'{name}[{i}]'] = variable[i].item()
        else:
            self.inner_state_debug[f'{name}'] = variable.item()



in_mult = 5
out_mult = 40

class KalmanNetNN_arch2(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self,learnable_ar_coeff,PCI_MODEL):
        super().__init__()
        self.learnable_ar_coeff = learnable_ar_coeff
        self.PCI = PCI_MODEL

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, SysModel):

        self.InitSystemDynamics(SysModel.F, SysModel.H, SysModel.m, SysModel.n, infoString = "partialInfo")

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S):

        self.seq_len_input = 1
        self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        if self.learnable_ar_coeff:
            self.p = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Q).to(dev, non_blocking=True)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).to(dev, non_blocking=True)

        # GRU to track S
        self.d_input_S = self.n ** 2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.seq_len_input, self.batch_size, self.d_hidden_S).to(dev, non_blocking=True)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

        """
        # Fully connected 8
        self.d_input_FC8 = self.d_hidden_Q
        self.d_output_FC8 = self.d_hidden_Q
        self.d_hidden_FC8 = self.d_hidden_Q * Q_Sigma_mult
        self.FC8 = nn.Sequential(
                nn.Linear(self.d_input_FC8, self.d_hidden_FC8),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC8, self.d_output_FC8))
        """

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H, m, n, infoString = 'fullInfo'):
        
        if(infoString == 'partialInfo'):
            self.fString ='ModInacc'
            self.hString ='ObsInacc'
        else:
            self.fString ='ModAcc'
            self.hString ='ObsAcc'
        
        # Set State Evolution Function
        self.F = F
        self.m = m



        # Set Observation Function
        self.H = H
        self.n = n

        ###################################
        ### Set if the NET is learnable ###
        ###################################
    def set_is_learnable(self, is_learnable):
        self.learnable = is_learnable
        for var in list(self.parameters()):
            var.requires_grad = is_learnable

    ########################################
    ### Set if the ar coeff is learnable ###
    ########################################
    def set_learnable_ar_coeff(self,is_learnable):
        self.learnable_ar_coeff=is_learnable
        self.p = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        self.T = T
        self.x_out = torch.empty(self.m, T).to(dev, non_blocking=True)
        self.m1x_posterior = M1_0.to(dev, non_blocking=True)
        self.m1x_posterior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.m1x_prior_previous = self.m1x_posterior.to(dev, non_blocking=True)
        self.y_previous = torch.matmul(self.H,self.m1x_posterior).to(dev, non_blocking=True)
        self.dy = 0

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T,self.m,self.n)).to(dev, non_blocking=True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)
        if self.learnable_ar_coeff and self.PCI:
            self.m1x_prior[-1]=  self.p * self.m1x_posterior[-1]
            # temp_p = self.p.detach()
            new_F=self.F.clone()
            new_F[-1,-1] = self.p
            self.F = new_F#self.p.clone().detach()
        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H.clone(),self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def     step_KGain_est(self, y):

        self.inner_state_debug = {}

        obs_diff = torch.squeeze(self.y_previous) /y
        obs_innov_diff =torch.squeeze(self.m1y) / y
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_posterior_previous)
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous)

        self.update_inner_state_debug('F1 obs_diff', obs_diff)
        self.update_inner_state_debug('F2 obs_innov_diff', obs_innov_diff)
        self.update_inner_state_debug('F3 fw_evol_diff', fw_evol_diff)
        self.update_inner_state_debug('F4 fw_update_diff', fw_update_diff)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix
        self.KGain = (torch.reshape(KG, (self.m, self.n)))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Save KGain in array
        self.KGain_array[self.i] = self.KGain
        self.i += 1

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y
        self.dy = dy

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.INOV=INOV
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior

        # update y_prev
        self.y_previous = y

        self.update_inner_state_debug('m1x prior', self.m1x_prior)
        self.update_inner_state_debug('m1y', self.m1y)
        self.update_inner_state_debug('dy', self.dy)
        self.update_inner_state_debug('innovation', self.INOV)
        self.update_inner_state_debug('m1x_posterior', self.m1x_posterior)





        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        self.update_inner_state_debug('F1 obs_diff_norm', obs_diff)
        self.update_inner_state_debug('F2 obs_innov_diff_norm', obs_innov_diff)
        self.update_inner_state_debug('F3 fw_evol_diff_norm', fw_evol_diff)
        self.update_inner_state_debug('F4 fw_update_diff_norm', fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)
        self.update_inner_state_debug('out_FC5', out_FC5)


        # Q-GRU
        in_Q = out_FC5

        out_Q,self.h_Q= self.GRU_Q(in_Q, self.h_Q.detach())
        self.update_inner_state_debug('out_Q', out_Q)


        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)
        self.update_inner_state_debug('out_FC6', out_FC6)


        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma.detach())
        self.update_inner_state_debug('out_Sigma', out_Sigma)


        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)
        self.update_inner_state_debug('out_FC1', out_FC1)


        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)
        self.update_inner_state_debug('out_FC7', out_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S,self.h_S = self.GRU_S(in_S, self.h_S.detach())
        self.update_inner_state_debug('out_S', out_S)



        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)


        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)
        self.update_inner_state_debug('out_FC3', out_FC3)


        # FC     4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
        self.update_inner_state_debug('out_FC4/h_Sigma', out_FC4)


        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        self.update_inner_state_debug('Gain', out_FC2)

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        # torch.autograd.set_detect_anomaly(True)
        y = y.to(dev, non_blocking=True)
        self.H[0,0] = y[0]
        y_beta = y[1:2].clone() #take beta asset
        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        self.x_out = self.KNet_step(y_beta)

        return self.x_out

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

    def update_inner_state_debug(self,name,variable):
        variable = variable.detach().squeeze().reshape(-1,1)
        if variable.shape[0]>1:
            for i in range(variable.shape[0]):
                self.inner_state_debug[f'{name}[{i}]'] = variable[i].item()
        else:
            self.inner_state_debug[f'{name}'] = variable.item()



