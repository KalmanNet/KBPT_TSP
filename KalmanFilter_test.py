import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter
import matplotlib.pyplot as plt
from positions import bollinger
from Utilities import *
from datetime import datetime
import os
from statsmodels.tsa.stattools import adfuller



from Extended_data import N_T

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")




def KFTest(SysModel, test_input, test_target,train_input,train_target,cv_input,cv_target,supervised,STD_WINDOW_SIZE,test_dates,KF_with_trainable_BB =False,BB_Version = 0,CO_INTEG = False,Synth_Data=False):
    # LOSS



    KF_data_saved = {}
    loss_fn = nn.MSELoss(reduction='mean')
    N_T = test_input.shape[0]
    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    # start = time.time()
    KF_train = KalmanFilter(SysModel)
    KF_cv = KalmanFilter(SysModel)
    KF = KalmanFilter(SysModel)



    if BB_Version==2:
        size_of_position_value=2
    else:
        size_of_position_value=2

    position_rolling_std = torch.zeros(1,size_of_position_value, KF.T_test)
    z_score_rolling_std = torch.zeros(N_T, 1, KF.T_test)
    rolling_var = torch.zeros(N_T, KF.T_test)


    for j in range(0, N_T):
        #### Test KF on non Trainable BB #####
        initial_value = get_initial_values(test_target,train_input,SysModel.m)#test_target[:, :, 0].T.type(torch.float32)
        # initial_value[1,:] = initial_value[1,:] * test_input[:,0,0]
        KF.InitSequence(initial_value, SysModel.m2x_0)
        KF.GenerateSequence(test_input[j, :, :], KF.T_test)


        if supervised==True:
            MSE_KF_linear_arr[j] = loss_fn(KF.x, test_target[j, :, :]).item()
        else:
            MSE_KF_linear_arr[j] = loss_fn(KF.y_vector, test_input[j, 1, :].reshape(1,-1)).item()

        dy = KF.dy_vector.reshape(-1)
        estimated_state_vector = KF.x


        position_rolling_std_hybrid = torch.zeros(2,size_of_position_value, KF.T_test)


        fig, ax = plt.subplots(1, 1)

        for i in [(.3,-0.3)]:

            BB_position = bollinger(is_learnable=False,version=BB_Version , short_TH=i[0],long_TH=i[1])
            BB_position.set_is_learnable(False)

            if CO_INTEG:
                indicator = estimated_state_vector[2,:]
            else:
                indicator = dy

            for t in range(STD_WINDOW_SIZE,KF.T_test):
                rolling_var[j,t] = torch.var(indicator[t-STD_WINDOW_SIZE:t]) #KF.m2y_vector[0,t]
                z_score_rolling_std[j, :, t] = indicator[t] / torch.sqrt(rolling_var[j,t])
                position_rolling_std_hybrid[Hybrid_mode_counter, :, t][0],position_rolling_std_hybrid[Hybrid_mode_counter, :, t][1]= position_rolling_std[j, :, t][0],position_rolling_std[j, :, t][1] =  \
                    BB_position(z_score_rolling_std[j, :, t], position_rolling_std[j, :, t - 1],is_soft=False)



            position_output = position_rolling_std

            # Delete opening position on last day
            position_output[j,0,-1]=0

            #close all open trades
            opening_days = np.where(position_output[j,0,:]!=0)[0]
            closing_days = np.where(position_output[j,1,:]!=0)[0]
            if len(opening_days) == len(closing_days) + 1:
                position_output[j, 1, -1] = position_output[j,0,opening_days[-1]]



            KF_train.InitSequence(get_initial_values(train_target,train_input,SysModel.m), SysModel.m2x_0)
            KF_train.GenerateSequence(test_input[j, :, :], KF_train.T_test)
            KF_cv.InitSequence(get_initial_values(cv_target,cv_input,SysModel.m), SysModel.m2x_0)
            KF_cv.GenerateSequence(cv_input[j, :, :], KF_cv.T_cv)

            cum_pnl_rolling_std,pnl_rolling_std = pnl_loss(position=position_rolling_std, assets=test_input[j, :, :],
                                           state_vectors=estimated_state_vector)

            cum_pnl_rolling_std=cum_pnl_rolling_std.detach()

            short_TH,long_TH = BB_position.get_THs() ##test_dates
            plot_data(ax,range(KF.T_test),cum_pnl_rolling_std, label=f'Prev DBB  LT: {long_TH} ST: {short_TH}')
            print(f"CUM PNL Prev DBB : {cum_pnl_rolling_std[-1]}")


            ax.set_title("KF PNL")
            ax.grid(True)

            ################
            ### Get Time ###
            ################
            today = datetime.today()
            now = datetime.now()
            strToday = today.strftime("%m_%d_%y")
            strNow = now.strftime("%H_%M_%S")
            strTime = strToday + "_" + strNow
    ax.legend()




    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)


    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")


    target_state_vector = test_target.squeeze().numpy()

    num_plots = 2
    if supervised == False:
        num_plots+=1
    if CO_INTEG:
        num_plots += 1

    fig, ax = plt.subplots(num_plots, 1)
    ax[0].plot(estimated_state_vector[0, :], label='estimated h')
    if Synth_Data:
        ax[0].plot(target_state_vector[0, :], label='real h')
    ax[0].legend()

    ax[1].plot(estimated_state_vector[1, :], label='estimated equilibrium')
    if Synth_Data:
        ax[1].plot(target_state_vector[1, :], label='real spread')
    ax[1].legend()


    if CO_INTEG:
        ax[2].plot(estimated_state_vector[2, :], label='estimated AR factor')
        if Synth_Data:
            ax[2].plot(target_state_vector[2, :], label='real AR Factor')
        ax[2].legend()

    if supervised == False:
        ax[-1].plot(KF.y_vector.reshape(-1), label='estimated beta asset')
        ax[-1].plot(test_input[:, 1, :].reshape(-1), label='real beta asset')
        ax[-1].legend()
    plt.suptitle(f"Kalman Filter (MSE {MSE_KF_dB_avg})")


    KF.position = position_output
    KF_data_saved['Test Set KF'] = KF
    KF_data_saved['Train Set KF'] = KF_train
    KF_data_saved['CV Set KF'] = KF_train

    investment_statistics(pnl_rolling_std[STD_WINDOW_SIZE:], position_output)

    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg ,cum_pnl_rolling_std,pnl_rolling_std,KF_data_saved]



