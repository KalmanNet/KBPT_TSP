import copy

import numpy as np
import torch
import torch.nn as nn
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from Utilities import pnl_loss,train_only_BB,get_initial_values,investment_statistics
from statsmodels.tsa.stattools import adfuller
import os


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")


class Pipeline_EKF:

    def __init__(self,pipeline_specifications, folderName,position_model_input):
        super().__init__()
        self.Time = pipeline_specifications['strTime']
        self.supervised=pipeline_specifications['supervised']
        self.pnl_loss_function = pipeline_specifications['pnl_loss_function']
        self.three_step_training = pipeline_specifications['three_step_training']
        self.loss_function_in_steps = pipeline_specifications['STEPS_LOSS_FUNC']
        self.position_model_input = position_model_input
        self.show_loss_in_DB=pipeline_specifications['show_loss_in_DB']
        self.STD_WINDOW_SIZE = pipeline_specifications['ROLLING_STD_WINDOW_SIZE']
        self.pipeline_name = pipeline_specifications['Pipeline_Name']
        self.Take_Last_Trained = pipeline_specifications['take_last_trained_model']
        self.folderName = folderName
        self.STEPS_TO_DO = pipeline_specifications['STEPS_TO_DO']
        ##For Legacy reasons, E2E is called Step 3
        if 2 in self.STEPS_TO_DO:
            self.STEPS_TO_DO[self.STEPS_TO_DO.index(2)] = 3

        self.BB_VERSION = pipeline_specifications['BB_VERSION']
        self.BB_learnable = pipeline_specifications['bollinger_learnable']
        self.STEPS_INFO = pipeline_specifications['STEPS_INFO']
        self.PCI = pipeline_specifications['PCI_Model']
        self.Learnable_AR_Coff = pipeline_specifications['Learnable_AR_coeff']
        self.Syn_Data = pipeline_specifications['Synthetic_Data']
        self.Informative_Test_Results = pipeline_specifications['Informative_Test_Results']

        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)


        self.CRASHED = False #For automation



    def save(self,name=None):
        if name is None:
            name = self.PipelineName
        torch.save(self, name)

    def setssModel(self, ssModel):
        self.ssModel = ssModel
        self.Data_set_Name = ssModel.DataSet_Name

    def setModel(self, model):
        if self.PCI:
            SS_MODEL = "PCI"
        else:
            SS_MODEL = "CI"
        self.model = model.to(dev, non_blocking=True)
        self.modelName = f"KNET_{SS_MODEL}"
        self.modelFileName = self.folderName + "/" + self.modelName + self.pipeline_name+".pt"
        self.modelFileName_editable = self.folderName + "/" +  self.modelName + self.pipeline_name #for 3 step trainning
        self.PipelineName = self.folderName + "/" + self.modelName +"_pipeline.pt"
    def setPositionModel(self, position_model):
        if self.PCI:
            SS_MODEL = "PCI"
        else:
            SS_MODEL = "CI"
        self.position_model = position_model.to(dev,non_blocking=True)
        self.position_model_name = position_model.__class__.__name__ +"_" + SS_MODEL
        self.PositionFileName = self.folderName + "/" +"POSITION___" + self.position_model_name + self.pipeline_name+ ".pt"
        self.PositionFileName_editable = self.folderName + "/" + "POSITION___" + self.position_model_name + self.pipeline_name #for 3 step trainning
        self.position_model.set_is_learnable(self.BB_learnable)



    def print_pipeline_info(self):
        print(f"\n\n###### PIPELINE INFO ########")

        print(f"Supervised - {self.supervised}")
        print(f"Position Model Input - {self.position_model_input}")
        print(f"3 Step Training - {self.three_step_training}")
        if not(self.three_step_training):
            print(f"Loss Function PNL - {self.pnl_loss_function}")
            if self.pnl_loss_function:
                print(f"Bollinger Learnable - {self.position_model.learnable}")
        else:
            for step, step_values in self.STEPS_INFO.items():
                print(step)
                for hyperparameter_name,hyperparameter_value in step_values.items():
                    print(f"\t{hyperparameter_name} : {hyperparameter_value}")



        print(f"Taking Last Trained Model - {self.Take_Last_Trained}")


        print("#############################\n\n")
    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.combined_model = nn.ModuleList([self.model, self.position_model])
        self.optimizer = torch.optim.Adam(self.combined_model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, train_input, train_target, cv_input, cv_target):
        cv_target = cv_target.to(dev, non_blocking=True)
        train_target = train_target.to(dev, non_blocking=True)


        if self.PCI:
            if self.position_model_input == 'dy':
                main = 'dy'
                alt = 'm_t'
            else:
                alt = 'dy'
                main = 'm_t'
        else:
            main = alt = 'dy'



        train_input_unbatched = train_input
        train_target_unbatched = train_target

        self.N_CV = cv_input.size()[0]

        num_steps = 3 if self.three_step_training else 1

        self.last_state_vector = torch.zeros(4,self.ssModel.m,1,requires_grad=False)


        # One Step Mode : KNET Supervised, KNET Unsupervised , KNET + BB End 2 End
        # Two Step Mode :
        #   1) KNET Unsupervied
        #   2) Train BB on Best KNET trained model TODO
        #   3) E2E
        for step in range(1, 1 + num_steps):


            if self.three_step_training and step == 1:
                if step not in self.STEPS_TO_DO:
                    continue
                print("\n\nTraining KNET : Unsupervised (MSE)\n\n")


            elif step == 2:
                if step not in self.STEPS_TO_DO:
                    continue
                print(f"\n\nTraining BB : {self.loss_function_in_steps[step]}\n\n")
                self.model = torch.load(f"{self.modelFileName_editable}_STEP1.pt",map_location=dev)  # take the best KNET model that was trained

                x_out_training_step_two = torch.empty(self.ssModel.m, self.ssModel.T_train,requires_grad=False).to(dev, non_blocking=True)
                dy_vec_train_step_two = torch.zeros(self.ssModel.T_train, device=dev, requires_grad=False)

                x_out_cv_step_two = torch.empty(self.ssModel.m, self.ssModel.T_cv, requires_grad=False).to(dev,non_blocking=True)
                dy_vec_cv_step_two = torch.zeros(self.ssModel.T_cv, device=dev, requires_grad=False)

                initial_value = get_initial_values(train_target_unbatched,train_input_unbatched,self.ssModel.m) # [H ratio , Spread]train_target[0, :, 0:1]
                self.model.InitSequence(initial_value, self.ssModel.T_train)
                for t in range(0, self.ssModel.T_train):
                    x_out_training_step_two[:, t] = self.model(train_input_unbatched[0,:, t])
                    dy_vec_train_step_two[t] = self.model.dy

                initial_value = get_initial_values(cv_target,cv_input,self.ssModel.m)  # [H ratio , Spread]
                self.model.InitSequence(initial_value, self.ssModel.T_cv)
                for t in range(0,self.ssModel.T_cv):
                    x_out_cv_step_two[:, t] = self.model(cv_input[0, :, t])
                    dy_vec_cv_step_two[t] = self.model.dy

                #no need for batches

                train_only_BB(train_input=train_input_unbatched.detach(),state_vector_train=x_out_training_step_two.detach(),dy_vector_train=dy_vec_train_step_two.detach(),
                          cv_input=cv_input.detach(),state_vector_cv=x_out_cv_step_two.detach(),dy_vector_cv=dy_vec_cv_step_two.detach(),STD_WINDOW_SIZE=self.STD_WINDOW_SIZE,
                          num_epochs=self.STEPS_INFO['STEP2']['epochs'],loss_function=self.loss_function_in_steps[step],
                          randomize_initial_values=False,save_model_as=f"{self.PositionFileName_editable}_STEP2.pt",
                          version=self.BB_VERSION,learning_rate=self.STEPS_INFO['STEP2']['lr'],pipeline_info=self)

                continue
            elif step == 3:
                if step not in self.STEPS_TO_DO:
                    continue
                print(f"\n\nTraining KNET  + BB : {self.loss_function_in_steps[step]}\n\n")
                #Load the best BB trained in step 2 [The best Knet Model is already loaded]
                self.position_model = torch.load(f"{self.PositionFileName_editable}_STEP2.pt",map_location=dev)
                self.model = torch.load(f"{self.modelFileName_editable}_STEP1.pt",map_location=dev)  # take the best KNET model that was trained


                self.N_Epochs = self.STEPS_INFO['STEP3']['epochs']
                self.N_B = self.STEPS_INFO['STEP3']['batches']
                self.pnl_loss_function = True
                self.show_loss_in_DB = not(self.pnl_loss_function)
                self.position_model.set_is_learnable(False)
                self.model.set_is_learnable(True)
                self.model.set_learnable_ar_coeff(False)
                self.setTrainingParams(n_Epochs=self.N_Epochs, n_Batch=self.N_B, learningRate=self.STEPS_INFO['STEP3']['lr'], weightDecay=1e-6)



            #Make Batches Info

            data_input = train_input_unbatched # train_input


            total_samples = data_input.size()[2]
            self.samples_per_batch = (total_samples // self.N_B)
            train_input = data_input[:, :, :self.samples_per_batch * self.N_B]
            train_target = train_target_unbatched[:, :, :self.samples_per_batch * self.N_B] # so it will divide equally
            temp_train_input = torch.zeros(self.N_B, train_input.size()[1], self.samples_per_batch)
            temp_train_target = torch.zeros(self.N_B, train_target.size()[1], self.samples_per_batch)

            # Reshape didnt do it right? fix it to be one line of code
            for b in range(self.N_B):
                if self.Syn_Data==True:
                    for i in range(train_target.size()[1]):
                        temp_train_target[b, i, :] = train_target[0, i,
                                                     b * self.samples_per_batch:(b + 1) * self.samples_per_batch]
                for i in range(train_input.size()[1]):
                    temp_train_input[b, i, :] = train_input[0, i,
                                                b * self.samples_per_batch:(b + 1) * self.samples_per_batch]

            train_input = temp_train_input
            train_target = temp_train_target

            print(f"Size of Train - {train_input.size()}")
            print(f"Size of CV - {cv_input.size()}\n\n")







            MSE_cv_linear_batch = torch.empty([self.N_CV],requires_grad=False).to(dev, non_blocking=True)
            self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs],requires_grad=False).to(dev, non_blocking=True)
            self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs],requires_grad=False).to(dev, non_blocking=True)

            MSE_train_linear_batch = torch.empty([self.N_B]).to(dev, non_blocking=True)
            self.MSE_train_linear_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)
            self.MSE_train_dB_epoch = torch.empty([self.N_Epochs]).to(dev, non_blocking=True)

            #########################
            ### Supervised Sanity ###
            #########################

            if self.supervised==True and train_target.size()[0]==0:
                print("ERROR - Supervised Mode w/o Target Data")
                exit(1)





            ##############
            ### Epochs ###
            ##############
            self.MSE_cv_opt = 1e10
            self.MSE_cv_idx_opt = 0
            self.optimal_short_TH=0;
            self.optimal_long_TH = 0;

            if self.BB_VERSION == 2:
                size_of_position_value = 2
            else:
                size_of_position_value = 1

            positions_cv = torch.zeros(self.N_Epochs, size_of_position_value, self.ssModel.T_cv, device=dev)
            positions_cv_alt = torch.zeros(self.N_Epochs, size_of_position_value, self.ssModel.T_cv, device=dev)
            z_score_cv = torch.zeros(self.N_Epochs, 1, self.ssModel.T_cv, device=dev)
            z_score_cv_alt= torch.zeros(self.N_Epochs, 1, self.ssModel.T_cv, device=dev)

            for ti in range(0, self.N_Epochs):

                #################################
                ### Validation Sequence Batch ###
                #################################
                # Cross Validation Mode
                self.model.eval()
                self.position_model.eval()
                self.position_model.reset_states()
                alt_position_model = copy.deepcopy(self.position_model)




                for j in range(0, self.N_CV):
                    y_cv = cv_input[j, :, :] # [Alpha_asset,Beta_asset]
                    initial_value=get_initial_values(cv_target,cv_input,self.ssModel.m) # [H ratio , Spread]
                    self.model.InitSequence(initial_value, self.ssModel.T_cv)

                    x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T_cv)
                    y_out_cv = torch.empty(self.ssModel.n, self.ssModel.T_cv)
                    dy_cv = torch.zeros(self.ssModel.T_cv)

                    for t in range(0, self.ssModel.T_cv):
                        x_out_cv[:, t] = self.model(y_cv[:, t])
                        y_out_cv[:, t] = self.model.m1y.squeeze()
                        dy_cv[t] = self.model.dy

                        if torch.isnan(dy_cv[t]) or torch.isinf(torch.abs(dy_cv[t])):
                            print(f'\nERROR in CV: dy = {dy_cv[t].item()}')
                            break


                        z_score_cv[ti, :, t] , value, var = self.get_position_model_input(state_vector=x_out_cv[:, max(0,t-self.STD_WINDOW_SIZE):t+1], assets=y_cv[:,max(0,t-self.STD_WINDOW_SIZE):t+1], dy_vec=dy_cv[max(0,t-self.STD_WINDOW_SIZE):t+1],pos_model_input=main)
                        z_score_cv_alt[ti, :, t] , value_alt , var_alt  = self.get_position_model_input(state_vector=x_out_cv[:, max(0,t-self.STD_WINDOW_SIZE):t+1], assets=y_cv[:,max(0,t-self.STD_WINDOW_SIZE):t+1], dy_vec=dy_cv[max(0,t-self.STD_WINDOW_SIZE):t+1],pos_model_input=alt)
                        if t>self.STD_WINDOW_SIZE: #decreases chances std is nan(atleast dy case)
                            positions_cv[ti, :, t][0],positions_cv[ti, :, t][1]= self.position_model(z_score_cv[ti, :, t],positions_cv[ti, :, t - 1].clone(),is_soft = False)
                            positions_cv_alt[ti, :, t][0],positions_cv_alt[ti, :, t][1]= alt_position_model(z_score_cv_alt[ti, :, t],positions_cv_alt[ti, :, t - 1].clone(),is_soft = False)

                    if self.pnl_loss_function == True:
                        cum_pnl,pnl = pnl_loss(position=positions_cv[ti, :, :], assets=y_cv, state_vectors=x_out_cv)
                        cum_pnl_alt,pnl_alt = pnl_loss(position=positions_cv_alt[ti, :, :], assets=y_cv, state_vectors=x_out_cv)

                        MSE_cv_linear_batch[j] = -cum_pnl[-1].item()
                    else:
                        if self.supervised == True:
                            MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv[:,:], cv_target[j, :, :]).item()
                        else:
                            MSE_cv_linear_batch[j] = self.loss_fn(y_out_cv[:, :], cv_input[j, 1, :].reshape(1,-1)).item()



                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.Take_Last_Trained or (self.show_loss_in_DB and self.MSE_cv_dB_epoch[ti] < self.MSE_cv_opt) or (not(self.show_loss_in_DB) and self.MSE_cv_linear_epoch[ti] < self.MSE_cv_opt)):
                    self.MSE_cv_opt = self.MSE_cv_dB_epoch[ti] if self.show_loss_in_DB else self.MSE_cv_linear_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    self.optimal_long_TH,self.optimal_short_TH = self.position_model.get_THs()
                    if self.three_step_training:
                        self.last_dys = dy_cv
                        torch.save(self.model, f"{self.modelFileName_editable}_STEP{step}.pt")
                        if step==3:
                            torch.save(self.position_model, f"{self.PositionFileName_editable}_STEP{step}.pt")

                    else:
                        torch.save(self.model,self.modelFileName)
                        if self.position_model.learnable:
                            torch.save(self.position_model, self.PositionFileName)


                    self.last_state_vector[step,:,:] = x_out_cv[:,-1].reshape(-1,1)
                    self.save()
    
                ###############################
                ### Training Sequence Batch ###
                ###############################

                # Training Mode
                self.model.train()
                self.position_model.train()
                # Init Hidden State
                self.model.init_hidden()

                Batch_Optimizing_LOSS_sum = 0


                for j in range(0, self.N_B):
                    # n_e = random.randint(0, self.N_E - 1)
                    # print(f"Batch {j}")
                    if step==3 and ti<3:
                        max_sample=200
                        self.samples_per_batch=200
                    else:
                        max_sample=train_input.size()[-1]
                        self.samples_per_batch = max_sample
                    y_training = train_input[j, :, :max_sample] #[Alpha_asset, Beta_asset]
                    initial_value = get_initial_values(train_target[j,:,:],y_training,self.ssModel.m)
                    self.position_model.reset_states()

                    # if j>1:
                    #     initial_value = (x_out_training[:,-1:].detach() + initial_value)/2
                    self.model.InitSequence(initial_value,self.ssModel.T_train)

                    x_out_training = torch.empty(self.ssModel.m, self.samples_per_batch).to(dev, non_blocking=True)
                    y_out_training = torch.empty(self.ssModel.n, self.samples_per_batch).to(dev, non_blocking=True)
                    positions_train_epoch = torch.zeros(1, size_of_position_value, self.samples_per_batch, device=dev)
                    z_score_train_epoch = torch.zeros(1, 1, self.samples_per_batch, device=dev)
                    dy_vec_train = torch.zeros(self.samples_per_batch, device=dev,requires_grad=False)

                    inner_states = []

                    for t in range(0, self.samples_per_batch):
                        x_out_training[:, t] = self.model(y_training[:, t])
                        y_out_training[:, t] = self.model.m1y.squeeze()
                        inner_states.append(self.model.inner_state_debug)
                        dy_vec_train[t]  = self.model.dy

                        if torch.isnan(dy_vec_train[t]) or torch.isinf(torch.abs(dy_vec_train[t])):
                            print(f'\nERROR : dy = {dy_vec_train[t].item()}')
                            df_inner_states = pd.DataFrame(inner_states)
                            # df_inner_states.to_csv(f'/Users/amitmilstein/Documents/Ben_Gurion_Univ/MSc/Thesis/My KalmanBOT/KNet/KNet_TSP/KNet/model/inner_state_debug_folder/CRASHED_{self.pipeline_name}_EPOCH{ti}.csv')
                            self.CRASHED=True
                            return

                        xvector = x_out_training[:, max(0,t-self.STD_WINDOW_SIZE):t+1]
                        alpha_vector = y_training[:, max(t-self.STD_WINDOW_SIZE,0):t+1]

                       #DEBUG purposes
                        # value_var_train[ti,0,t] = value
                        # value_var_train[ti,1,t] = var
                        if t>self.STD_WINDOW_SIZE:
                            z_score_train_epoch[:, :, t], value, var = self.get_position_model_input(
                                state_vector=x_out_training[:, max(0, t - self.STD_WINDOW_SIZE):t + 1],
                                assets=y_training[:, max(t - self.STD_WINDOW_SIZE, 0):t + 1],
                                dy_vec=dy_vec_train[max(t - self.STD_WINDOW_SIZE, 0):t + 1])

                            positions_train_epoch[0,:, t][0],positions_train_epoch[0,:, t][1] = self.position_model(z_score_train_epoch[:,:,t], positions_train_epoch[:,:, t - 1].clone(),is_soft = True)




                    if self.pnl_loss_function==True :
                        cum_pnl, pnl = pnl_loss(position=positions_train_epoch, assets=y_training,
                                                state_vectors=x_out_training)


                        if self.loss_function_in_steps[step] == 'max_P':
                            maximize_profit = torch.cumsum(pnl[pnl>0],dim=0)
                            if len(maximize_profit)>0:
                                LOSS = - maximize_profit[-1]
                            else:
                                LOSS = - cum_pnl[-1]
                        if self.loss_function_in_steps[step] == 'max_PNL' or self.three_step_training==False:
                            LOSS = - cum_pnl[-1]

                    else:
                        if self.supervised == True:
                            LOSS = self.loss_fn(x_out_training[:,:self.samples_per_batch], train_target[j, :, :self.samples_per_batch])
                        else:
                            LOSS = self.loss_fn(y_out_training[:, :], y_training[1, :].reshape(1,-1))


                    MSE_train_linear_batch[j] = LOSS.item()




                    ##################
                    ### Optimizing ###
                    ##################

                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable
                    # weights of the model). This is because by default, gradients are
                    # accumulated in buffers( i.e, not overwritten) whenever .backward()
                    # is called. Checkout docs of torch.autograd.backward for more details.
                    self.optimizer.zero_grad()

                    LOSS.backward(retain_graph=True)

                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    self.optimizer.step()

                # Average
                self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
                self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])



                if ti==0 and False:
                    df_inner_states = pd.DataFrame(inner_states)
                    df_inner_states.to_csv(f'{os.getcwd()}/KNet/KNet_TSP/KNet/model/inner_state_debug_folder/{self.ssModel.DataSet_Name}_{self.modelName}_EPOCH{ti}.csv')

                # print(f'\n\nUPDATED KNET : {torch.any(KNET_b4!=KNET_AFTER)}\nUPDATED POSITION: {torch.any(POSITION_b4!=POSITION_AFTER)}\n\n')



                ########################
                ### Training Summary ###
                ########################
                if self.show_loss_in_DB: #Loss in DB
                    LOSS_FUNCTION="MSE"
                    error_train = self.MSE_train_dB_epoch
                    error_cv = self.MSE_cv_dB_epoch
                    units = "[dB]"
                else:
                    LOSS_FUNCTION = "-1*PNL"
                    error_train = self.MSE_train_linear_epoch
                    error_cv = self.MSE_cv_linear_epoch
                    units = f""
                if self.PCI:
                    PCI_INFO = 'ar coeff : ' ,self.model.F[-1,-1].item()
                else:
                    PCI_INFO = ""

                Epoch_short_TH , Epoch_Long_TH = self.position_model.get_THs()
                print(ti,LOSS_FUNCTION,"Training :", error_train[ti], units,LOSS_FUNCTION, "Validation :", error_cv[ti],
                      units, "Long TH : " , Epoch_Long_TH,"Short TH : ",Epoch_short_TH , PCI_INFO)

                if (ti > 1):
                    d_train = error_train[ti] - error_train[ti - 1]
                    d_cv = error_cv[ti] - error_cv[ti - 1]
                    print("diff",LOSS_FUNCTION,"Training :", d_train, units, "diff",LOSS_FUNCTION, "Validation :", d_cv, units)

                print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_opt, units)

            # train_MSE = error_train
            # validation_MSE = error_cv
            # plt.figure()
            # plt.plot(train_MSE, label='Training')
            # plt.plot(validation_MSE, label='Validation')
            # plt.legend()
            # plt.ylabel(f'{LOSS_FUNCTION} {units}')
            # plt.title("Learning Curve")



        

    def NNTest(self, test_input, test_target,test_dates,info_string="",KF_estimated_state = None,KF_positions=None): #KF data for debug
        test_target = test_target.to(dev, non_blocking=True)
        self.N_T = test_input.size()[0]
        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        if self.PCI:
            if self.position_model_input == 'dy':
                main = 'dy'
                alt = 'm_t'
            else:
                alt = 'dy'
                main = 'm_t'
        else:
            main = alt = 'dy'



        self.model.eval()
        self.position_model.eval()
        self.position_model.reset_states()

        position_model_alt = copy.deepcopy(self.position_model)

        if self.BB_VERSION == 2:
            size_of_position_value = 2
        else:
            size_of_position_value = 1

        if self.PCI == False:
            alt_input = 'dy'
        else:
            alt_input = 'm_t'

        torch.no_grad()

        x_out_array = torch.empty(self.N_T,self.ssModel.m, self.ssModel.T_test)
        y_out_array = torch.empty(self.N_T,self.ssModel.n, self.ssModel.T_test)
        estimated_spread = torch.empty(self.N_T, 1 , self.ssModel.T_test)
        positions_test = torch.zeros(self.N_T, size_of_position_value, self.ssModel.T_test, device=dev)
        positions_alt_test = torch.zeros(self.N_T, size_of_position_value, self.ssModel.T_test, device=dev)

        z_score_test = torch.zeros(self.N_T, 1, self.ssModel.T_test, device=dev)
        z_score_alt_test = torch.zeros(self.N_T, 1, self.ssModel.T_test, device=dev)

        cum_pnl = torch.zeros(self.N_T, self.ssModel.T_test, device=dev)

        start = time.time()

        # #For testing
        # self.last_state_vector=self.last_state_vector.detach()
        # self.last_state_vector[0, :, :] = test_target[0,:,0:1]


        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]
            if self.three_step_training:
                initial_value =  self.last_state_vector[3, :,:]
            else:
                initial_value = self.last_state_vector[1, :, :]  #

            self.model.InitSequence(initial_value, self.ssModel.T_test)

            x_out_test = torch.empty(self.ssModel.m,self.ssModel.T_test,requires_grad=False)
            y_out_test = torch.empty(self.ssModel.n,self.ssModel.T_test,requires_grad=False)
            estimated_spread_test = torch.empty(1,self.ssModel.T_test,requires_grad=False)

            dy_test = torch.empty(self.ssModel.T_test)

            for t in range(0, self.ssModel.T_test):

                x_out_test[:, t] = self.model(y_mdl_tst[:, t])
                dy_test[t] = self.model.dy
                y_out_test[:, t] = self.model.m1y.squeeze()

                estimated_spread_test[:,t] = x_out_test[1, t]

                z_score_test[j, :, t],value,var = self.get_position_model_input(state_vector=x_out_test[:,max(t-self.STD_WINDOW_SIZE,0):t+1],assets=y_mdl_tst[:, max(t-self.STD_WINDOW_SIZE,0):t+1],dy_vec=dy_test[max(t-self.STD_WINDOW_SIZE,0):t+1],pos_model_input=main)
                z_score_alt_test[j, :, t],value_alt,var_alt = self.get_position_model_input(state_vector=x_out_test[:,max(t-self.STD_WINDOW_SIZE,0):t+1],assets=y_mdl_tst[:, max(t-self.STD_WINDOW_SIZE,0):t+1],dy_vec=dy_test[max(t-self.STD_WINDOW_SIZE,0):t+1],pos_model_input=alt)

                if t > self.STD_WINDOW_SIZE:
                    positions_test[j, :, t][0],positions_test[j, :, t][1] = self.position_model(z_score_test[j, :, t], positions_test[j, :, t - 1],is_soft=False)
                    positions_alt_test[j, :, t][0],positions_alt_test[j, :, t][1] = position_model_alt(z_score_alt_test[j, :, t], positions_test[j, :, t - 1],is_soft=False)


            cum_pnl_epoch,pnl_epoch = pnl_loss(position=positions_test[j, :, :], assets=y_mdl_tst, state_vectors=x_out_test)
            cum_pnl_epoch_alt, pnl_epoch_alt = pnl_loss(position=positions_alt_test[j, :, :], assets=y_mdl_tst,state_vectors=x_out_test)



            cum_pnl[j] = cum_pnl_epoch.reshape(-1)
            if True or self.pnl_loss_function==False: #always so MSE error
                if self.supervised==True:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(y_out_test, y_mdl_tst[1, :].reshape(1,-1)).item()
            else: #PNL LOSS FUNCTION
                self.MSE_test_linear_arr[j] = -cum_pnl[j][-1]


            x_out_array[j,:,:] = x_out_test
            y_out_array[j,:,:] = y_out_test
            estimated_spread[j,:,:] = estimated_spread_test
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # H =x_out_test.detach().numpy()[0,:]
        # alpha = test_input.squeeze()[0,:].numpy()
        # beta = test_input.squeeze()[1,:].numpy()
        # plt.figure
        # plt.plot(alpha * H)
        # plt.plot(beta)
        # plt.plot(alpha)
        # plt.legend(['alpha*H','beta','alpha'])
        # plt.show()


        # # Standard deviation
        # self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        # self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

        # Print MSE Cross Validation
        str = self.modelName + " - " + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]" , )
        print(f"{self.modelName} - MSE: {self.MSE_test_dB_avg.item()} , PNL {main} - {cum_pnl_epoch.reshape(-1).detach()[-1]},PNL {alt} - {cum_pnl_epoch_alt.reshape(-1).detach()[-1]}")


        # Print Run Time
        print("Inference Time:", t)

        num_plots=2
        if self.supervised==False:
            num_plots+=1
            str_supervised = 'unsupervised'
        else:
            str_supervised = 'supervised'

        if self.PCI:
            num_plots += 1

        str_supervised = info_string

        if self.Informative_Test_Results: #plots
            # self.position_model.plot_THs(info_string)
            fig, ax = plt.subplots(num_plots, 1)
            ax[0].plot(x_out_test.detach()[0, :], label=r'$\hat{h}_t$')
            if self.Syn_Data:
                ax[0].plot(test_target.squeeze()[0, :], label='real h')
            ax[0].legend()

            ax[1].plot(estimated_spread.squeeze().detach(), label=r'$\hat{\mu}_t$')
            if self.Syn_Data:
                ax[1].plot(test_target.squeeze()[1, :], label='real spread')
            ax[1].legend()

            if self.PCI:
                ax[2].plot(x_out_test.detach()[2, :], label=r'$\hat{m}_t$')
                if self.Syn_Data:
                    ax[2].plot(test_target.squeeze()[2, :], label='real ar model')
                ax[2].legend()
                # plt.figure()
                # plt.plot(x_out_test.detach()[2, :]-0.05, label='estimated ar model')
                # plt.plot(test_target.squeeze()[2, :], label='real ar model')
                # plt.legend()

            if self.supervised == False:
                ax[-1].plot(y_out_test.detach().reshape(-1), label=r'$\hat{\beta}_t$')
                ax[-1].plot(y_mdl_tst[1, :], label=r'$\beta_t$')
                ax[-1].legend()
            plt.suptitle(f"KBOT {str_supervised} (MSE {round(self.MSE_test_dB_avg.item(),2)} [dB]) \n {self.Data_set_Name}")
        Test_Short_TH, Test_Long_TH = self.position_model.get_THs()
        print(f"TEST : Long TH : {Test_Long_TH}, Short TH : {Test_Short_TH}")


        torch.save(self.model, os.path.join(self.folderName,'KNET_PCI_POST_TEST.pt'))
        with torch.no_grad():
            self.last_state_vector[0, :, :] = x_out_test[:,-1].detach().reshape(-1, 1) #Save the last state space vector
            self.last_dys = dy_test
        self.save(os.path.join(self.folderName,'KNET_pipeline_POST_TEST.pt'))

        results_main = \
            {'pnl':pnl_epoch.reshape(-1).detach(),
             'cum_pnl':cum_pnl_epoch.reshape(-1).detach(),
             'positions':positions_test.reshape(-1).detach(),
             'state_vectors' : x_out_test.detach(),
             'assets': y_mdl_tst.detach(),
             'beta_estimated' : y_out_test.detach().reshape(-1),
             'beta_real' : y_mdl_tst[1, :]
            }
        results_alt = \
            {'pnl': pnl_epoch_alt.reshape(-1).detach(),
             'cum_pnl': cum_pnl_epoch_alt.reshape(-1).detach(),
             'positions': positions_alt_test.reshape(-1).detach(),
             'state_vectors': x_out_test.detach(),
             'assets': y_mdl_tst.detach(),
             'beta_estimated': y_out_test.detach().reshape(-1),
             'beta_real': y_mdl_tst[1, :]
             }

        investment_statistics(results_main['pnl'][self.STD_WINDOW_SIZE:],positions_test)


        return [results_main,results_alt]

    def get_position_model_input(self,state_vector,assets,dy_vec,pos_model_input=None):
        if pos_model_input==None:
            pos_model_input = self.position_model_input

        if pos_model_input == 'dy':
            mean_dy = 0#   torch.mean(dy_vec)
            value  = value_dy = self.model.dy - mean_dy
            std = std_dy =  torch.std(dy_vec.detach()) #torch.tensor(0.002**2) #
            z_score = value_dy / std_dy

        if pos_model_input == 'm_t':
            mean_ar_model = 0#torch.mean(state_vector[2, :])
            std = torch.std(state_vector[2, :]).detach()
            value = state_vector[2, -1] - mean_ar_model
            z_score = value / std




        if pos_model_input == 'spread': #bad name
            size_of_current_LB = assets.size()[1]
            spread_with_alpha =state_vector[1,-size_of_current_LB:] * assets[0,:]
            mean = torch.mean(spread_with_alpha)
            value = spread_with_alpha[-1] - mean
            std = torch.std(spread_with_alpha)
            z_score = value / std

        if pos_model_input == 'h_ratio':  # bad name
            size_of_current_LB = assets.size()[1]
            #the beta asset we have is normalized by the alpha asset (data generator)
            spread_v2 = assets[0,:]*assets[1,:] - state_vector[0,-size_of_current_LB:]*assets[0,:]
            mean = torch.mean(spread_v2)
            value = spread_v2[-1] - mean
            std = torch.std(spread_v2)
            z_score = value/std

        return z_score , value ,std


