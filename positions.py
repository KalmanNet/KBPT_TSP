import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from torch.nn.modules.linear import Linear

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

LONG_POSITION = 1
SHORT_POSITION = -1
HOLD_POSITION = 0
CLOSE_POSITION = 1

class bollinger(torch.nn.Module):
    def __init__(self, std = 1,is_learnable=True ,short_TH=1./3,long_TH=-1./3, version = 2):
                                                    # 0.2     -0.3
        super().__init__()
        self.device = dev
        self.to(self.device)
        self.std = std


        self.version = version

        if version == 0 or version == 2:
            self.short_TH = nn.Parameter(torch.tensor([short_TH]), requires_grad=True)
            self.long_TH = nn.Parameter(torch.tensor([long_TH]), requires_grad=True)
            self.TH_sets = 1
            self.open_position_indicator= 0 # Positive-->True , Negative-->False
            self.open_position = 0
            self.prev_z = None
            if version==2:
                self.close_THs = nn.Parameter(torch.tensor([0.,0.]), requires_grad=True)

        if version == 1:
            if not(isinstance(short_TH, list)):
                short_TH= [short_TH,short_TH,short_TH]
            if not(isinstance(long_TH, list)):
                long_TH = [long_TH, long_TH, long_TH]

            self.short_TH_prev_pos_hold = nn.Parameter(torch.tensor([short_TH[0]]), requires_grad=True)
            self.long_TH_prev_post_hold = nn.Parameter(torch.tensor([long_TH[0]]), requires_grad=True)
            self.short_TH_prev_pos_long = nn.Parameter(torch.tensor([short_TH[1]]), requires_grad=True)
            self.long_TH_prev_pos_long = nn.Parameter(torch.tensor([long_TH[1]]), requires_grad=True)
            self.short_TH_prev_pos_short = nn.Parameter(torch.tensor([short_TH[2]]), requires_grad=True)
            self.long_TH_prev_pos_short = nn.Parameter(torch.tensor([long_TH[2]]), requires_grad=True)
            self.TH_sets = 3




        self.learnable = is_learnable
        self.set_is_learnable(self.learnable)


    def get_THs_internal(self):
        if self.version == 1 :
            current_short_TH = [self.short_TH_prev_pos_short, self.short_TH_prev_pos_hold, self.short_TH_prev_pos_long]
            current_long_TH = [self.long_TH_prev_pos_short, self.long_TH_prev_post_hold, self.long_TH_prev_pos_long]
        if self.version == 0 or self.version == 2:
            current_long_TH = [self.long_TH]
            current_short_TH = [self.short_TH]

        return  current_short_TH,current_long_TH

    def get_THs(self):
        short_THs = []
        long_THs = []

        current_short_TH, current_long_TH = self.get_THs_internal()

        for th in current_short_TH:
            short_THs.append(round(th.detach().item(), 2))
        for th in current_long_TH:
            long_THs.append(round(th.detach().item(), 2))

        return short_THs,long_THs


    def plot_THs(self,title):
        short_TH , long_TH = self.get_THs()
        last_post = [-1,0,1]
        fig,ax = plt.subplots(1,self.TH_sets)

        if self.TH_sets >1:
            for i in range(self.TH_sets):
                ax[i].axhline(y=short_TH[i], color='r', linestyle='-',label=short_TH[i])
                ax[i].axhline(y=long_TH[i], color='g', linestyle='-',label=long_TH[i])
                ax[i].set_xlabel(f"last pos {last_post[i]}")
                ax[i].set_ylim((-3,3))
                ax[i].legend()
        else:
            ax.axhline(y=short_TH[0], color='r', linestyle='-', label=short_TH[0])
            ax.axhline(y=long_TH[0], color='g', linestyle='-', label=long_TH[0])
            ax.set_ylim((-3, 3))
            ax.legend()
        plt.suptitle(f'{title} - BB THs')


    def get_regularization_factor(self):
        r=torch.tensor(0)
        short_THs, long_THs = self.get_THs_internal()
        for th in short_THs:
            r=r + torch.exp(-100 * th)
        for th in long_THs:
            r = r + torch.exp(100 * th)

        return r


    def set_is_learnable(self,is_learnable):
        self.learnable = is_learnable
        for var in list(self.parameters()):
            var.requires_grad=is_learnable

    def reset_states(self):
        if self.version==2:
            self.open_position_indicator= 0 # Positive-->True , Negative-->False
            self.open_position = 0
            self.prev_z = None


    def indicator_prev_hold(self, last_pos):
        if self.soft:
            return torch.exp(-1/(self.std**2*2)*((last_pos)**2))
        else:
            return last_pos==0

    def indicator_prev_long(self, last_pos):
        if self.soft:
            return torch.exp(-1/(self.std**2*2)*((last_pos-1)**2))
        else:
            return last_pos==1
        
    def indicator_prev_short(self, last_pos):
        if self.soft:
            return torch.exp(-1/(self.std**2*2)*((last_pos+1)**2))
        else:
            return last_pos==-1

    def u(self, arg):
        if self.soft:
            # return torch.nn.functional.relu(arg)
            return torch.distributions.normal.Normal(0,self.std, validate_args=None).cdf(arg)
        else:
            return torch.heaviside(arg,torch.tensor(0,dtype=arg.dtype)) # @ 0 return 0



    def forward(self, z, last_pos , aggressive = 1,is_soft = True):
        self.soft = is_soft

        if self.version == 0:
            a = LONG_POSITION*self.u((self.long_TH - z)) + SHORT_POSITION * self.u((z-self.short_TH)) # z>self.short_TH --> -1 or z<self.long_TH ---> 1
            b = 0
            c = 0

        if self.version == 1:
            a = (LONG_POSITION*self.u(self.long_TH_prev_post_hold - z) + SHORT_POSITION * self.u(z - self.short_TH_prev_pos_hold)) * self.indicator_prev_hold(last_pos)  # IF last_post ==0 z>self.short_TH --> -1 or z<self.long_TH ---> 1
            b = (LONG_POSITION*self.u(self.long_TH_prev_pos_long - z) + SHORT_POSITION * self.u(z - self.short_TH_prev_pos_long)) * self.indicator_prev_long(last_pos)
            c = (LONG_POSITION*self.u(self.long_TH_prev_pos_short - z) + SHORT_POSITION * self.u(z - self.short_TH_prev_pos_short)) * self.indicator_prev_short(last_pos)

        if self.version == 2:
            if self.prev_z==None:
                self.prev_z=z.detach()

            #Should I Close Position?
            close_position_indicator = CLOSE_POSITION * self.u(-self.prev_z*z) * self.open_position_indicator#If your holding something, close it. b==8 --> close position
            # close_position_indicator = torch.round(close_position_indicator)


            # this tells us if we closed a short or long. if its -1 we close a short if its 1 we closed a long
            # Needed for easy PNL calculation
            close_position = close_position_indicator * self.open_position

            close_position_indicator_rounded = torch.round(close_position_indicator)

            #Do i pass a TH that suggests i should take a position
            take_position_first_criteria = (LONG_POSITION*self.u(self.long_TH - z)) + SHORT_POSITION * self.u(z-self.short_TH) # z>self.short_TH --> -1 or z<self.long_TH ---> 1)
            # take_position_first_criteria = torch.round(take_position_first_criteria)


            # 'close_position_indicator' will never equal '1 - self.open_position_indicator'
            # if close_position_indicator==1 --> self.open_position_indicator=1
            # but if self.open_position_indicator==1 -/-> close_position_indicator=1
            # if self.open_position_indicator==0 --> close_position_indicator=0
            # but close_position_indicator==0 -/-> self.open_position_indicator=0
            take_position_decision = take_position_first_criteria * (close_position_indicator_rounded + (1-self.open_position_indicator))
            take_position_decision_rounded=torch.round(take_position_decision)

            #Update States

            prev_open_position_indicator = self.open_position_indicator
            self.open_position = (prev_open_position_indicator * (close_position_indicator_rounded*take_position_decision_rounded + (1-close_position_indicator_rounded)*self.open_position) + (1-prev_open_position_indicator) * take_position_decision_rounded).detach()
            self.open_position_indicator = (torch.round(self.open_position)**2)
            self.prev_z=z.detach()


            return take_position_decision,close_position




        return (a+b+c),0



