import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN_arch2 , KalmanNetNN_arch1
from matplotlib.backends.backend_pdf import PdfPages
from KalmanNet_sysmdl import SystemModel
import copy
# import datetime;
from KalmanFilter_test import KFTest
import matplotlib.pyplot as plt
import datetime
from Utilities import *


def multipage(filename, figs=None, dpi=200):
   ct = datetime.now()
   time_stamp_string = f"{ct.day}_{ct.month}_{ct.year}_{ct.hour}H_{ct.minute}M"
   pp = PdfPages(filename + "_" + time_stamp_string+".pdf")
   if figs is None:
      figs = [plt.figure(n) for n in plt.get_fignums()]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()



if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

   

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
# print("Current Time =", strTime)
path_results = 'KNet/'





# import_data('CHF_EURO')#EWC_EWA CHF_EURO AUD_ZAR


for index in range(0,1):


   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataSetFolderName = r'DataSet/'
   dataSetFileName = 'CHF_EURO.pt' ## EWC_EWA-A CHF_EURO AUD_ZAR

   dataSetName = dataSetFileName.split('.')[0]
   dataFolderName = r'Saved Data/'+dataSetName+r'/' #where to save plots and such
   Synthetic_Data = True if 'generated' in dataSetFileName else False

   if not os.path.exists(dataFolderName):
      os.makedirs(dataFolderName)


   if torch.cuda.is_available():
      [train_dates, train_input, train_target, cv_dates, cv_input, cv_target, test_dates, test_input, test_target] = DataLoader_GPU(dataSetFolderName + dataSetFileName)
   else:
      [train_dates,train_input, train_target,cv_dates, cv_input, cv_target,test_dates, test_input, test_target] = DataLoader(dataSetFolderName + dataSetFileName)




   T_train = train_input.size()[-1]
   T_cv = cv_input.size()[-1]
   T_test = test_input.size()[-1]

   plot_basic_info_on_dataset(dataSetName,train_input,test_input,train_dates,test_dates)






   #####################
   ###     Config    ###
   #####################
   Train_KNET = False
   supervised = False and Synthetic_Data # for Synthetic Data
   three_step_training = True  # Train Knet unsupervised with MSE loss, then train separately the BB
   STEPS_TO_DO = [3]
   STEPS_LOSS_FUNC = ['NOT_USED', 'Unsupervised', 'max_PNL', 'max_PNL']#['NOT_USED','Unsupervised','max_P','max_PNL']
   pnl_loss_function = False and not (three_step_training)  # for end to end KNET + BB
   bollinger_learnable = False and pnl_loss_function  # for KNET only end to end --- only when pnl_loss_function=True
   KF_with_trainable_BB = False
   BB_VERSION = 2
   PCI_MODEL = False
   Learnable_AR_coeff = False and PCI_MODEL
   Informative_Test_Results = True
   Compare_Between_Steps = False

   SS_MODEL = "PCI" if PCI_MODEL else "CI"



   take_last_trained_model = False
   show_loss_in_DB = not (pnl_loss_function)
   position_model_input = 'dy' if PCI_MODEL==True else 'dy'

   STEP1_BATCHES_LIST = {"EWC_EWA-A": 20,"EWC_EWA-B":20, "CHF_EURO": 20,"AUD_ZAR":10}

   NUM_BATCHES_S1 = STEP1_BATCHES_LIST[dataSetName]
   NUM_BATCHES_S3 = 1

   STEP1_EPOCH_LIST = {"EWC_EWA-A": 30 , "EWC_EWA-B":20, "CHF_EURO": 15,"AUD_ZAR":10}
   STEP3_EPOCH_LIST = {"PCI":{"EWC_EWA-A": 18, "EWC_EWA-B":66,"CHF_EURO": 25,"AUD_ZAR":57},"CI":{"EWC_EWA-A": 31, "EWC_EWA-B":66,"CHF_EURO": 47,"AUD_ZAR":56}}

   NUM_EPOCHS = STEP1_EPOCH_LIST[dataSetName]  # This is for Step 1 As well
   NUM_EPOCHS_S2 = 1
   NUM_EPOCHS_S3 = STEP3_EPOCH_LIST[SS_MODEL][dataSetName]
   STEP1_LR_LIST = {"EWC_EWA-A": 9e-6,"EWC_EWA-B":4e-5, "CHF_EURO":1e-3 ,"AUD_CAD":1e-3,"AUD_ZAR":5e-4}
   STEP3_LR_LIST = {"PCI" : {"EWC_EWA-A": 3e-5,"EWC_EWA-B":3e-5, "CHF_EURO":1e-3,"AUD_CAD":9e-5,"AUD_ZAR":2e-4},"CI" : {"EWC_EWA-A": 3e-4,"EWC_EWA-B":3e-4, "CHF_EURO":1e-3,"AUD_CAD":9e-5,"AUD_ZAR":6e-4}}


   LR = STEP1_LR_LIST[dataSetName] # 1e-3
   LR_S2 = 8e-3
   LR_S3 = STEP3_LR_LIST[SS_MODEL][dataSetName]

   ROLLING_STD_WINDOW_SIZE = 80

   model_info = "__"
   if three_step_training:
      model_info += "three_step_training"
   else:
      if pnl_loss_function:
         model_info += "__PNL_LOSS"
         if bollinger_learnable:
            model_info += "__DBB"
         else:
            model_info += "__normal_BB"
      else:
         model_info += "__MSE_LOSS"
         if supervised:
            model_info += "__supervised"
         else:
            model_info += "__unsupervised"

   Step_Info = {
      'STEP1': {'lr': LR, 'epochs': NUM_EPOCHS ,"batches" : NUM_BATCHES_S1},
      'STEP2': {'lr': LR_S2, 'epochs': NUM_EPOCHS_S2, "batches" : 1},
      'STEP3': {'lr': LR_S3, 'epochs': NUM_EPOCHS_S3,"batches" : NUM_BATCHES_S3}
   }

   Pipeline_Specifications = {
      'supervised': supervised,
      'three_step_training': three_step_training,
      'STEPS_LOSS_FUNC' : STEPS_LOSS_FUNC,
      'pnl_loss_function': pnl_loss_function,
      'bollinger_learnable': bollinger_learnable,
      'KF_with_trainable_BB': KF_with_trainable_BB,
      'take_last_trained_model': take_last_trained_model,
      'show_loss_in_DB': show_loss_in_DB,
      'ROLLING_STD_WINDOW_SIZE': ROLLING_STD_WINDOW_SIZE,
      'Pipeline_Name': model_info,
      'strTime': strTime,
      'STEPS_TO_DO': STEPS_TO_DO,
      'BB_VERSION': BB_VERSION,
      'STEPS_INFO': Step_Info,
      'PCI_Model' : PCI_MODEL,
      'Learnable_AR_coeff' : Learnable_AR_coeff,
      'Synthetic_Data' : Synthetic_Data,
      'Data_Set_Name': dataSetName,
      'Informative_Test_Results': Informative_Test_Results
   }


   #####################
   ###  Design Model ###
   #####################


   if PCI_MODEL:
      size_of_SS_vector = 3


   else:
      size_of_SS_vector = 2
      p = 1

   m1x_0 = torch.zeros((size_of_SS_vector, 1))
   m2x_0 = torch.zeros((size_of_SS_vector, size_of_SS_vector))



   q_list = {"CI": {"EWC_EWA-A": 0.009,"EWC_EWA-B":0.009, "CHF_EURO":0.2 ,"AUD_ZAR":0.194}, "PCI" : {"EWC_EWA-A": 0.0105,"EWC_EWA-B":0.008, "CHF_EURO":0.29 ,"AUD_ZAR":0.21}}
   r_list = {"CI": {"EWC_EWA-A": 0.64,"EWC_EWA-B":0.64, "CHF_EURO":0.898 ,"AUD_ZAR":0.7}, "PCI" : {"EWC_EWA-A": 0.81 ,"EWC_EWA-B":0.56, "CHF_EURO":1.11,"AUD_ZAR":1}}
   p_list = {"EWC_EWA-A": 0.1, "EWC_EWA-B": 0.01, "CHF_EURO": 0.17, "AUD_ZAR": 0.17}

   q =q_list["PCI" if PCI_MODEL else"CI"][dataSetName] #0.002
   r = r_list["PCI" if PCI_MODEL else"CI"][dataSetName]#0.0009000000427477062/1#0.002
   p = p_list[dataSetName]

   q_check_list = np.arange(0.1,1,0.1) #np.arange(0.01,.11,0.01)
   r_check_list = np.arange(0.1,1,.1)
   p_check_list = np.arange(0.1,1,0.1)

   best_pnl=-1
   best_q=0
   best_r=0
   best_p = 0
   for q in [q]:
      for r in [r]:
         for p in [0]:
            F = torch.eye(size_of_SS_vector)
            if PCI_MODEL:
               F[-1,-1] = p #if no Co-Integrated Model p has no meaning

            H = torch.ones(size_of_SS_vector, dtype=torch.float32).reshape(1, -1)

            Q_true =torch.eye(size_of_SS_vector) * q**2
            if PCI_MODEL:
               Q_true[-1,-1] = 0.035**2


            R_true = r ** 2 * torch.eye(1)


            sys_model = SystemModel(F, Q_true, H, R_true, T_train,T_cv, T_test,dataSetName)
            sys_model.InitSequence(m1x_0, m2x_0)

            if Synthetic_Data:
               train_target=train_target[:,:size_of_SS_vector,:]
               cv_target=cv_target[:,:size_of_SS_vector,:]
               test_target=test_target[:,:size_of_SS_vector,:]







            ######################
            ###  Kalman Filter ###
            ######################

            KF_sys_model = copy.deepcopy(sys_model)
            [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,KF_cum_pnl,KF_pnl,KF_data_saved] = KFTest(KF_with_trainable_BB=KF_with_trainable_BB,SysModel=KF_sys_model, test_input=test_input, test_target=test_target,train_input=train_input,train_target=train_target,cv_input=cv_input,cv_target=cv_target,supervised=supervised, STD_WINDOW_SIZE=ROLLING_STD_WINDOW_SIZE,BB_Version = BB_VERSION,CO_INTEG = PCI_MODEL,Synth_Data=Synthetic_Data,test_dates=train_dates)
            if best_pnl<KF_cum_pnl[-1].item():
               best_pnl = KF_cum_pnl[-1].item()
               best_q = q
               best_r = r
               best_p = p

   print(f"Best PNL: {best_pnl}, q : {best_q} , r : {best_r} p {best_p}")

   ##################
   ###  KalmanNet ###
   ##################
   sys_model.F[-1, -1] = 0.8 #The Models ive trained all used 0.8 as p ratio
   modelFolder = 'Models' + '/' + dataSetName
   KNet_Pipeline = Pipeline_EKF(pipeline_specifications=Pipeline_Specifications, folderName=modelFolder, position_model_input = position_model_input)
   KNet_Pipeline.setssModel(sys_model)
   KNet_model = KalmanNetNN_arch2(Learnable_AR_coeff,PCI_MODEL)
   KNet_model.Build(sys_model)

   position_model = bollinger(is_learnable=Pipeline_Specifications['bollinger_learnable'],version=Pipeline_Specifications['BB_VERSION'])
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setPositionModel(position_model)
   KNet_Pipeline.print_pipeline_info()
   KNet_Pipeline.setTrainingParams(n_Epochs=Step_Info['STEP1']['epochs'], n_Batch=Step_Info['STEP1']['batches'], learningRate=Step_Info['STEP1']['lr'], weightDecay=5e-4)

   #Save the KF results
   try:
      np.save(dataFolderName + give_name(KNet_Pipeline, 'KF_pnl'), KF_cum_pnl)
   except:
      print("No KF PNL")
   if Train_KNET:
      KNet_Pipeline.NNTrain(train_input, train_target, cv_input, cv_target)
      if KNet_Pipeline.CRASHED:
         exit(1)
   else:
      KNet_Pipeline = torch.load(KNet_Pipeline.PipelineName)
      KNet_Pipeline.Informative_Test_Results = Informative_Test_Results



   KNet_Pipeline.STD_WINDOW_SIZE = 60

   if three_step_training == False:


      if KNet_Pipeline.position_model.learnable:  # If i learned a specific BB , load it . Else, just go with the standard
         KNet_Pipeline.position_model = torch.load(KNet_Pipeline.PositionFileName, map_location=dev)

      [KNET,KNET_alt] = KNet_Pipeline.NNTest(test_input, test_target,test_dates=test_dates)

      np.save(dataFolderName + give_name(KNet_Pipeline, 'KNET_pnl'), KNET['cum_pnl'])


      KNET_cum_pnl_PCI = load_data(dataFolderName + 'PCI__KNET_cum_pnl.npy')
      KNET_cum_pnl_CI = load_data(dataFolderName + 'CI__KNET_cum_pnl.npy')
      KF_cum_pnl_co_PCI = load_data(dataFolderName + 'PCI__KF_cum_pnl.npy')
      KF_cum_pnl_CI = load_data(dataFolderName + 'CI__KF_cum_pnl.npy')


      fig,ax = plt.subplots()
      plot_data(ax,test_dates,KNET_cum_pnl_CI,label=f'KNET CI')
      plot_data(ax,test_dates,KNET_cum_pnl_PCI,label=f'KNET PCI')
      plot_data(ax,test_dates,KF_cum_pnl_CI,label=f'KF CI')
      plot_data(ax,test_dates,KF_cum_pnl_co_PCI,label=f'KF PCI')
      ax.set_ylabel('USD')
      ax.set_xlabel('Date')
      plt.grid(True)


      # plt.plot(KF_cum_pnl, label='KF')
      plt.legend(fontsize=LEGEND_FONT_SIZE)
      plt.title(f"PNL {index}")

   else:



      if Compare_Between_Steps: #compare in between steps
         fix_main, ax_main = plt.subplots(1, 1)
         fig_track_comp, ax_track_comp = plt.subplots(1, 1)

         for window_size in [60]:
            KNet_Pipeline.STD_WINDOW_SIZE = window_size
            #First do the STEP 1 KNET with the STEP 2 BB
            print("STEP 1 KNET Step 2 BB")
            KNet_Pipeline.model = torch.load(f"{KNet_Pipeline.modelFileName_editable}_STEP1.pt", map_location=dev) #load the best KNET that was trained
            KNet_Pipeline.position_model = torch.load(f"{KNet_Pipeline.PositionFileName_editable}_STEP2.pt", map_location=dev)
            [KNET_step1__position_step2,KNET_step1__position_step2_alt] = KNet_Pipeline.NNTest(test_input, test_target,test_dates=test_dates,info_string="Training Step 1")


            # STEP 2 KNET with the STEP 2 BB
            print("STEP 2 KNET Step 2 BB")
            KNet_Pipeline.model = torch.load(f"{KNet_Pipeline.modelFileName_editable}_STEP3.pt",map_location=dev)  # load the best KNET that was trained
            KNet_Pipeline.position_model = torch.load(f"{KNet_Pipeline.PositionFileName_editable}_STEP2.pt", map_location=dev)
            [KNET_step2__position_step2,KNET_step2__position_step2_alt] = KNet_Pipeline.NNTest(test_input, test_target,test_dates=test_dates,info_string="Training Step 2")


            plot_data(ax_track_comp,test_dates,(KNET_step2__position_step2['beta_real']-KNET_step1__position_step2['beta_estimated']),label=r'$\beta_t -\hat{\beta}_t^{Step1}$',linewidth=1,color=(1, 0, 0))
            plot_data(ax_track_comp,test_dates,(KNET_step2__position_step2['beta_real']-KNET_step2__position_step2['beta_estimated']),label=r'$\beta_t - \hat{\beta}_t^{Step2}$',linewidth=1,linestyle='-.',color=(0.4, 0, 0.4))
            ax_track_comp.set_ylabel('Price [USD]',fontsize=LABEL_FONT_SIZE)
            ax_track_comp.set_xlabel('Date',fontsize=LABEL_FONT_SIZE)
            ax_track_comp.legend(fontsize=LEGEND_FONT_SIZE)
            ax_track_comp.grid(True)

            plot_data(ax_main,test_dates,KNET_step1__position_step2['cum_pnl'],linestyle='-.',color=(1, 0, 0), label=f'Step 1')
            plot_data(ax_main,test_dates,KNET_step2__position_step2['cum_pnl'],linestyle='--',color=(0, 0.4, 0), label=f'Step 2')
            ax_main.set_ylabel('PNL [USD]',fontsize=LABEL_FONT_SIZE)
            ax_main.set_xlabel('Date',fontsize=LABEL_FONT_SIZE)
            ax_main.legend(fontsize=LEGEND_FONT_SIZE)
            ax_main.grid(True)

         if BB_VERSION!=2:
            list_of_models = [KNET_step3__position_step3_alt,KNET_step3__position_step3]
            hybrid_cum_pnl = hybrid_model(list_of_models)

      else:

         KNet_Pipeline.model = torch.load(f"{KNet_Pipeline.modelFileName_editable}_STEP3.pt",
                                          map_location=dev)  # load the best KNET that was trained
         KNet_Pipeline.position_model = torch.load(f"{KNet_Pipeline.PositionFileName_editable}_STEP3.pt",
                                                   map_location=dev)
         [KNET_step3__position_step3, KNET_step3__position_step3_alt] = KNet_Pipeline.NNTest(test_input, test_target,test_dates=test_dates,
                                                                        info_string="KNETS3 DBBS3")

         if KNet_Pipeline.position_model_input=='m_t':
            main='mt'
            alt = 'dy'
         else:
            main = 'dy'
            alt = 'mt'

         np.save(dataFolderName + give_name(KNet_Pipeline, 'KNET_pnl',f'_{alt}'), KNET_step3__position_step3_alt['cum_pnl'])
         np.save(dataFolderName + give_name(KNet_Pipeline, 'KNET_pnl',f'_{main}'), KNET_step3__position_step3['cum_pnl'])

         KNET_cum_pnl_PCI_mt = load_data(dataFolderName + 'PCI__3_step_training_mt__KNET_cum_pnl.npy')
         KNET_cum_pnl_PCI_dy = load_data(dataFolderName + 'PCI__3_step_training_dy__KNET_cum_pnl.npy')
         KNET_cum_pnl_CI_dy = load_data(dataFolderName + 'CI__3_step_training_dy__KNET_cum_pnl.npy')
         DDQN_cum_pnl = load_data(f'/Users/amitmilstein/Documents/Ben_Gurion_Univ/MSc/Thesis/Reinforcement_Learning/Saved_Data/{dataSetName}/cum_pnl_DDQN.npy')


         KF_cum_pnl_PCI = load_data(dataFolderName + 'PCI__KF_cum_pnl.npy')
         KF_cum_pnl_CI = load_data(dataFolderName + 'CI__KF_cum_pnl.npy')

         fig, ax = plt.subplots()
         plot_data(ax,test_dates,KF_cum_pnl_CI, label='B1', linestyle=(0, (5, 5)),color= (0, 0, 0.5)) #KF CI
         plot_data(ax,test_dates,KF_cum_pnl_PCI, label='B2', linestyle=":",color= (0.4, 0.2, 0))#KF PCI
         plot_data(ax, test_dates, KNET_cum_pnl_CI_dy, label=f'B3',linewidth=2, linestyle=(0, (3, 1)),color=(0, 0.4, 0)) #KNET CI
         plot_data(ax,test_dates,DDQN_cum_pnl,label = 'B4',linestyle='-.',color=(1, 0, 0)) #DDQN
         plot_data(ax,test_dates,KNET_cum_pnl_PCI_dy, label=f'Algorithm 1' , linestyle='-',color=(0.4, 0, 0.4)) #KNET PCI
         # plot_data(ax,test_dates,KNET_cum_pnl_PCI_mt, label=f'KNET PCI m_t')
         print(f"KF CI {KF_cum_pnl_CI.reshape(-1)[-1]}")
         print(f"KF PCI {KF_cum_pnl_PCI.reshape(-1)[-1]}")
         print(f"KNET CI {KNET_cum_pnl_CI_dy.reshape(-1)[-1]}")
         print(f"DDQN {DDQN_cum_pnl.reshape(-1)[-1]}")
         print(f"KNET PCI {KNET_cum_pnl_PCI_dy.reshape(-1)[-1]}")





         ax.set_ylabel('PNL [USD]',fontsize=LABEL_FONT_SIZE)
         ax.set_xlabel('Date',fontsize=LABEL_FONT_SIZE)
         plt.legend(fontsize=LEGEND_FONT_SIZE)
         plt.grid(True)

multipage(f"{os.getcwd()}/{dataFolderName}/{index}_{KNet_Pipeline.modelName}")
plt.show()