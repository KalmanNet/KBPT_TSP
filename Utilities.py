import torch
import numpy as np
from positions import bollinger
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import os
from datetime import datetime
import matplotlib.dates as mdates



LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


def pnl_loss( position, assets, state_vectors):
        return pnl_loss_with_closing_pos(position, assets, state_vectors)


def pnl_loss_original( position, assets, state_vectors):
    # We assume the following model --> asset[-1] = < h_vector , assets[:-1] > (in the case there are a pair of assets
    # then h_vector is of size 1
    # torch.autograd.set_detect_anomaly(True)
    positions = position[:,0,:].reshape(1, -1)

    pnl = torch.zeros_like(positions).T
    ones_vector = torch.ones_like(positions).T
    h_ratio_vector = state_vectors[0, :].reshape(-1,1)
    positions_duplicated = torch.cat((positions, positions)).T  # duplicate position vector
    h_one_vector = torch.cat((-h_ratio_vector, ones_vector),
                                axis=1)
    smart_poisitons = h_one_vector * positions_duplicated  # This tell us how many to buy sell of each asset
    asset_prices = assets.T

    buying_value = torch.sum(torch.abs(h_one_vector) * assets.T,axis=1)[:-1].reshape(-1,1)

    asset_prices_diff = torch.diff(asset_prices, axis=0)
    pnl[1:] = torch.sum(asset_prices_diff * smart_poisitons[:-1], axis=1).reshape(-1, 1)/buying_value

    cum_pnl = pnl
    cum_pnl = torch.cumsum(cum_pnl, dim=0).reshape(-1)### i change added the reshape 3/3
    return cum_pnl, pnl

def pnl_loss_with_closing_pos( position, assets, state_vectors):
    # We assume the following model --> asset[-1] = < h_vector , assets[:-1] > (in the case there are a pair of assets
    # then h_vector is of size 1
    # torch.autograd.set_detect_anomaly(True)



    positions = position.reshape(1,2,-1) #make it invariant to size (2,X) or (1,2,X)


    closing_positions = torch.round(positions[:,1,:].detach())
    opening_positions = torch.round(positions[:, 0, :].detach())


    #Delete Opening position on last sample if there is
    opening_positions[:,-1] = 0



    #Close all Open Positions
    mismatch = torch.sum(closing_positions == 0).item() - torch.sum(opening_positions == 0)
    if mismatch ==1:
        last_opening_position = opening_positions[:,torch.nonzero(opening_positions)[:,1][-1]]
        closing_positions[:,-1] = last_opening_position
        positions[:, 1, -1] = last_opening_position
    elif mismatch!=0:
        a=1






    closing_idx  = torch.nonzero(closing_positions.reshape(-1))[:,0]
    opening_idx  = torch.nonzero(opening_positions.reshape(-1))[:,0]



    pnl = torch.zeros_like(closing_positions).T
    ones_vector = torch.ones_like(closing_positions).T
    h_ratio_vector = state_vectors[0, :].reshape(-1,1)


    h_one_vector =  torch.cat((-h_ratio_vector, ones_vector),axis=1)

    normalize_assets = h_one_vector * assets.T
    normalized_alpha = -1*normalize_assets[:,0].detach().numpy() #redo the -1
    normalized_beta = normalize_assets[:, 1].detach().numpy()
    diff = normalized_beta-normalized_alpha


    derivative_value  = torch.sum(h_one_vector * assets.T,axis=1)
    buying_value = torch.sum(torch.abs(h_one_vector) * assets.T,axis=1)


    bad_values = (torch.isnan(derivative_value) | torch.isinf(torch.abs(derivative_value)))
    # #which values effect our PNL
    bad_values_with_positions = (bad_values[closing_idx] | bad_values[opening_idx])
    # #Remove the those positions
    bad_idx_closing = closing_idx[bad_values_with_positions]
    bad_idx_opening = opening_idx[bad_values_with_positions]
    closing_positions[:,bad_idx_closing] = 0
    opening_positions[:,bad_idx_opening] = 0
    #remove nans and infs
    derivative_value[bad_values] = 0 #Needed?

    #update idxs
    closing_idx = torch.nonzero(closing_positions.reshape(-1))[:, 0]
    opening_idx = torch.nonzero(opening_positions.reshape(-1))[:, 0]

    ones_vector = torch.ones_like(opening_idx).reshape(-1,1)

    opening_h_one_vector =  torch.cat((-h_ratio_vector[opening_idx], ones_vector),axis=1)
    closing_h_one_vector =  torch.cat((-h_ratio_vector[closing_idx], ones_vector),axis=1)

    opening_derivative_values_new = torch.sum(opening_h_one_vector * assets[:,opening_idx].T,axis=1)
    closing_derivative_values_new = torch.sum(closing_h_one_vector * assets[:,closing_idx].T,axis=1)

    buying_value = buying_value * torch.abs(opening_positions)
    buying_value = buying_value[:, opening_idx]

    closing_positions_unspaced=closing_positions[closing_positions!=0]

    Pnl_unspaced = (closing_derivative_values_new-opening_derivative_values_new) * closing_positions_unspaced
    Pnl_unspaced_normalized = Pnl_unspaced / buying_value
    pnl[closing_idx] = Pnl_unspaced_normalized.reshape(-1, 1)
    cum_pnl = torch.cumsum(pnl, dim=0)




    return cum_pnl, pnl

def pnl_loss_v1( position, assets, state_vectors):
    #This version we buy if assets always when they are over the TH, and only sell when the opposite position is taken

    # We assume the following model --> asset[-1] = < h_vector , assets[:-1] > (in the case there are a pair of assets
    # then h_vector is of size 1
    # torch.autograd.set_detect_anomaly(True)
    positions = position.reshape(-1)

    pnl = torch.zeros_like(positions).T
    ones_vector = torch.ones_like(positions).T.reshape(-1,1)
    h_ratio_vector = state_vectors[0, :].reshape(-1,1)
    price_of_derivative = torch.sum(torch.cat((-h_ratio_vector, ones_vector),axis=1) * assets.T,axis=1) #price of the "spread"

    last_position_trend = -1
    started=False
    num_derivatives_bought = 0
    value_of_bought_derivatives = 0
    for t in range(positions.size()[0]):
        if (started and positions[t]==-last_position_trend) or (not(started) and torch.abs(positions[t])==1):

            if started: #compute profit
                pnl[t] = last_position_trend * (num_derivatives_bought*price_of_derivative[t] - value_of_bought_derivatives)
                value_of_bought_derivatives = 0
                num_derivatives_bought = 0
            else:
                started=True

            last_position_trend = positions[t] #we are now in a new position trend
            continue


        if started and positions[t]==last_position_trend:
            num_derivatives_bought= num_derivatives_bought + 1
            value_of_bought_derivatives = value_of_bought_derivatives + price_of_derivative[t]

    cum_pnl = pnl
    cum_pnl = torch.cumsum(cum_pnl, dim=0).reshape(-1)### i change added the reshape 3/3
    return cum_pnl, pnl

def give_name(pipe_line,what,additional_flags=""):
    string = ""
    if 'pnl' in what:
        if pipe_line.PCI:
            string += "PCI"
        else:
            string += "CI"

        if pipe_line.three_step_training and 'KNET' in what:
            string += "__3_step_training"

        string += additional_flags + "__" + what.split('_')[0]+"_cum_pnl"

    return string


def get_initial_values(target_data,input_data,size_of_state_vector,mode=""):
    #Selects the way to compute the initial values



    if mode == "true" and torch.numel(target_data):
        # Sometimes data is in shape of (X,Y) and sometimes (1,X,Y) - this makes sure its always (1,X,Y)
        target_data = target_data.reshape(1, target_data.size()[-2], target_data.size()[-1])

        #reak initial values
        real_initial_value = target_data[0, :, 0:1]

        return real_initial_value
    elif mode == "zero":
        #initial values 0
        zero_intial_value = torch.zeros(size_of_state_vector, 1)

        return zero_intial_value
    else: ##Default

        # Sometimes data is in shape of (X,Y) and sometimes (1,X,Y) - this makes sure its always (1,X,Y)
        input_data = input_data.reshape(1, input_data.size()[-2], input_data.size()[-1])

        # Compute LR from data for initial values
        LR_intial_value = torch.zeros(size_of_state_vector, 1)
        p = np.polyfit(input_data[0, 0, :], input_data[0, 1, :], 1)
        LR_intial_value[0] = p[0]
        LR_intial_value[1] = p[1]
        # LR_intial_value[2] = input_data[0,1,0] - p[0]*input_data[0,0,0] - p[1] #beta_asset - alpha_asset * h[0] - d[0]

        return LR_intial_value


def import_data(data_csv=None,prices='Open'):
    #The Last col is the estimated asset, and all the ones before are the hedged assets



    dataFolderName='DataSet' + '/'
    name_of_assets=data_csv.split('-')[0].split('_')
    num_assets=len(name_of_assets)
    asset_dict ={}

    for i in range(num_assets):
        asset_dict[i]=name_of_assets[i]

    if data_csv=='CHF_EURO':
        train_size = 2000
        cv_size = 2000
        cv_train_size = 2000

        df = pd.read_csv(dataFolderName + data_csv + '.csv', index_col='time')
        wanted_alpha_column = df.columns.get_loc(f'chf')
        wanted_beta_column = df.columns.get_loc(f'eur')
        df_assets_data = df[df.columns[[wanted_alpha_column, wanted_beta_column]]]
        date_format = '%Y-%m-%d'


    elif 'generated' in data_csv:
        train_size = 2000
        cv_size = 500
        cv_train_size = cv_size + train_size

        df = pd.read_csv(dataFolderName + data_csv + '.csv')
        wanted_alpha_column = df.columns.get_loc(f'alpha_asset')
        wanted_beta_column = df.columns.get_loc(f'beta_asset')
        df_assets_data = df[df.columns[[wanted_alpha_column, wanted_beta_column]]]

        wanted_h_column = df.columns.get_loc(f'h_ratio')
        wanted_spread_ar_column = df.columns.get_loc(f'ar_proc')
        wanted_spread_column = df.columns.get_loc(f'spread')
        true_state_vector = df[df.columns[[wanted_h_column, wanted_spread_column,wanted_spread_ar_column]]]
        true_state_vector_train = true_state_vector.iloc[:train_size]
        true_state_vector_train = true_state_vector_train.to_numpy().T.reshape(1,-1,train.shape[-1])
        true_state_vector_test = true_state_vector.iloc[train_size:]
        true_state_vector_test = true_state_vector_test.to_numpy().T.reshape(1,-1,test.shape[-1])

        train_target = torch.tensor(true_state_vector_train,dtype=torch.float32)[:,:,:train_size]
        cv_target = torch.tensor(true_state_vector_train,dtype=torch.float32)[:,:,-cv_size:]
        test_target = torch.tensor(true_state_vector_test,dtype=torch.float32)


    else:
        #Yahoo Finance Data
        train_size = 2000
        cv_size = 2000
        cv_train_size = 2000
        alpha_asset_name = name_of_assets[0]
        beta_asset_name = name_of_assets[1]
        df = pd.read_csv(dataFolderName + data_csv + '.csv',index_col='Date')
        df = df.dropna(subset=df.columns[df.isnull().any()])
        df = df.iloc[-3500:,:]
        wanted_alpha_column = df.columns.get_loc(f'{alpha_asset_name} Open')
        wanted_beta_column = df.columns.get_loc(f'{beta_asset_name} Open')
        df_assets_data = df[df.columns[[wanted_alpha_column, wanted_beta_column]]]
        date_format = '%d/%m/%Y'


    train_df = df_assets_data.iloc[:cv_train_size]
    train = train_df.to_numpy().T.reshape(1,-1,len(train_df))
    test_df = df_assets_data.iloc[cv_train_size:]
    test = test_df.to_numpy().T.reshape(1,-1,len(test_df))


    train_input = torch.tensor(train,dtype=torch.float32)[:,:,:train_size]
    cv_input = torch.tensor(train,dtype=torch.float32)[:,:,-cv_size:]
    test_input = torch.tensor(test,dtype=torch.float32)






    if 'generated' not in data_csv:
        train_target = torch.empty(train_input.shape)
        cv_target = torch.empty(cv_input.shape)
        test_target = torch.empty(test_input.shape)

        # Take %Y of each sample for plot
        train_dates =np.array([datetime.strptime(date_str[:10], date_format) for date_str in train_df.axes[0].values[:train_size].astype(str)])
        cv_dates =np.array([datetime.strptime(date_str[:10], date_format) for date_str in train_df.axes[0].values[-cv_size:].astype(str)])
        test_dates =np.array([datetime.strptime(date_str[:10], date_format) for date_str in test_df.axes[0].values.astype(str)])

    else:
        train_dates = np.arange(train_input.shape[-1])
        cv_dates = np.arange(cv_input.shape[-1])
        test_dates = np.arange(test_input.shape[-1])


    torch.save([train_dates,train_input, train_target,cv_dates, cv_input, cv_target,test_dates, test_input, test_target], f'{os.getcwd()}/DataSet/{data_csv}.pt')


def hybrid_cum_pnl():
    return


def load_data(data_to_load):
    try:
        return np.load(data_to_load)
    except:
        print(f"{data_to_load} not found")
        return []

def plot_data(ax_object,x_axis,data,label,linestyle='-',linewidth=1.5,color=(0, 0.5, 0)):
    if len(data) > 0:
        ax_object.plot(x_axis,data,label=label,linewidth=linewidth,linestyle=linestyle,color=color)
        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter('%Y')
        ax_object.xaxis.set_major_locator(years)
        ax_object.xaxis.set_major_formatter(years_fmt)


def hybrid_model(list_of_models):
    T = len(list_of_models[0]['pnl'])
    num_of_models =len(list_of_models)
    look_back_period = 5
    Hybrid_positions = torch.zeros(1, T)


    look_back_cum_pnl=np.zeros((num_of_models,T))

    list_of_pnls=[]
    list_of_cum_pnls = []
    list_of_positions=[]

    for i in range(num_of_models):
        list_of_pnls.append(list_of_models[i]['pnl'].numpy())
        list_of_cum_pnls.append(list_of_models[i]['cum_pnl'].numpy())
        list_of_positions.append(list_of_models[i]['positions'])
        for t in range(0,T):
            look_back_cum_pnl[i,t] = np.sum(list_of_cum_pnls[i][max(t-look_back_period,0):t])


    best_model = np.argmax(look_back_cum_pnl,axis=0)
    for t in range(0,T):#Make more Efficient
        Hybrid_positions[:,t] = list_of_positions[best_model[t]][t]


    assets = list_of_models[0]['assets']
    state_vectors = list_of_models[0]['state_vectors']
    cum_pnl_hybrid,hybrid_pnl =pnl_loss(Hybrid_positions,assets,state_vectors)

    plt.figure()
    for i in range(num_of_models):
        plt.scatter(range(T),list_of_models[i]['cum_pnl'],s=2,label=f'{i}')
    plt.scatter(range(T),cum_pnl_hybrid,s=2,label='Hybrid')
    plt.legend()
    # plt.show()
    return cum_pnl_hybrid


def plot_basic_info_on_dataset(dataSetName,train_input,test_input,train_dates,test_dates):
    fig, ax = plt.subplots(2, 1)
    ax[0].scatter((train_input[:, 0, :]), (train_input[:, 1, :]), s=1, color='green', label='train')
    ax[0].set_ylabel(f"{dataSetName.split('_')[1]}")
    ax[0].legend()
    ax[1].scatter((test_input[:, 0, :]), (test_input[:, 1, :]), s=1, label='test')
    ax[1].set_xlabel(f"{dataSetName.split('_')[0]}")
    ax[1].set_ylabel(f"{dataSetName.split('_')[1]}")
    ax[1].legend()
    plt.suptitle(f"Linearity - {dataSetName}")
    fig, ax = plt.subplots(2, 1)
    plot_data(ax[0], train_dates, train_input[:, 0, :].reshape(-1), label=f"train alpha ({dataSetName.split('_')[0]})")
    plot_data(ax[0], train_dates, train_input[:, 1, :].reshape(-1), label=f"train beta ({dataSetName.split('_')[1]})")
    ax[0].legend()
    ax[0].set_ylabel(f"Price")
    plot_data(ax[1], test_dates, test_input[:, 0, :].reshape(-1), label=f"test alpha ({dataSetName.split('_')[0]})")
    plot_data(ax[1], test_dates, test_input[:, 1, :].reshape(-1), label=f"test beta ({dataSetName.split('_')[1]})")
    ax[1].set_xlabel(f"Year")
    ax[1].set_ylabel(f"Price")
    ax[1].legend()
    plt.suptitle(f"{dataSetName}")


def investment_statistics(pnl,pos=None):
    pnl=pnl.numpy()

    if pos is not None:

        pos=pos.squeeze()

        # Delete opening position on last day
        pos[0, -1] = 0

        # close all open trades
        opening_days = np.where(pos[0, :] != 0)[0]
        closing_days = np.where(pos[1, :] != 0)[0]
        if len(opening_days) == len(closing_days) + 1:
            pos[ 1, -1] = pos[0, opening_days[-1]]

        opening_days = np.where(pos[0, :] != 0)[0]
        closing_days = np.where(pos[1, :] != 0)[0]
        holding_times = closing_days - opening_days
        average_holding = np.mean(holding_times)
    else:
        average_holding=-1

    pnl_on_active_trades = pnl[pnl != 0]

    number_of_trade = len(pnl_on_active_trades)
    average_return = np.mean(pnl_on_active_trades)










    closing_trade_days = np.where(pnl != 0)[0]
    time_between_trading_days = np.diff(closing_trade_days)
    average_time_between_return = np.mean(time_between_trading_days)

    print(f"# trades {number_of_trade}")
    print(f"average return {average_return}")
    print(f"average holding {average_holding}")
    print(f"average time between return {average_time_between_return}")













