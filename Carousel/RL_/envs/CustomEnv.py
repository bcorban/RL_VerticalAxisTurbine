import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
import sys
import torch
import math
import gclib
import matlab.engine
import time
from scipy import signal
from scipy.io import savemat
from config import CONFIG_ENV

sys.path.append("/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs")

from param_matlab import param,m,NI

eng = matlab.engine.start_matlab()
path = "/Users/PIVUSER/Desktop/tmp_baptiste/Carousel/RL_/envs"
eng.addpath (path, nargout= 0 )
path = "/Users/PIVUSER/Documents/MATLAB"
eng.addpath(path, nargout= 0 )
g=gclib.py()
c=g.GCommand
g.GOpen('192.168.255.200 --direct -s ALL')
print(g.GInfo())

class CustomEnv(Env):
    
    def __init__(self) -> None:
        super(CustomEnv, self).__init__()
 
        self.action_space = Box(low=CONFIG_ENV['action_lb'], high=CONFIG_ENV['action_hb'], shape=(1,))
        self.observation_space = Box(low=np.array([-10,-10]), high=np.array([10,10]))       

        self.N_ep_without_homing=CONFIG_ENV['N_ep_without_homing']
        self.N_transient_effects=CONFIG_ENV['N_transient_effects']
        self.N_max=CONFIG_ENV['N_max']

        self.episode_counter=0

    def step(self, action): #perform one step, ie one pitching command
        
        self.pitch(action)
        
        self.read_state()
        
        self.history_states[self.i]=self.state
        self.history_action[self.i]=action
        
        nrot = self.history_phase_cont[self.i] // 360 #change index to get history_phase
        
        
        #Compute reward
        if nrot <= self.N_transient_effects + self.n_rot_ini:
            self.reward=0
            info={'transient':True}
        else:
            self.reward=(self.state[1]+2)/4 #-min/(max-min)
            info={'transient':False}
            
        #Check if episode terminated 
        if nrot>=self.N_max:
            terminated = True
            self.pitch(0) #pitch back to 0 at the end of an episode
        else:
            terminated = False

        truncated =False

        
        # Return step information
        return self.state, self.reward, terminated, truncated, info
    

    def reset(self,seed=None,options=None):
        self.episode_counter+=1
        N=100000
        self.history_phase=np.zeros(N)
        self.history_phase_cont=np.zeros(N)
        self.history_pitch_should=np.zeros(N)
        self.history_pitch_is=np.zeros(N)
        self.history_states=np.zeros(N,4)
        self.history_volts=np.zeros(N,8)
        self.history_volts_raw=np.zeros(N,8)
        self.history_time=np.zeros,(N)
        self.history_forces_noisy=np.zeros(N,2)
        self.history_forces_butter=np.zeros(N,2)
        self.history_forces=np.zeros(N,2)
        self.history_coeff=np.zeros(N,2)
        self.history_action=np.zeros(N)
        self.i=0
        
        if self.episode_counter>1: # if not first episode
            eng.stop_lc(nargout=0)
            self.save_data()
            
            if self.episode_counter%self.N_ep_without_homing==0: #if homing needed
                print("homing")
                self.stop_E()
                self.home()
                eng.start_lc(nargout=0) #Start the loadcell
                self.t_start=time.time()
                self.get_offset()
                self.start_E()
                self.wait_N_rot(3)
            else:
                eng.start_lc(nargout=0) #Start the loadcell
                self.t_start=time.time()
        
        else: #first episode : start load cell, get offset, start motor
            eng.start_lc(nargout=0) #Start the loadcell
            self.t_start=time.time()
            self.get_offset()
            self.start_E()
            self.wait_N_rot(3)

            
        info={}
        #-----------------------------------
        
       
        self.read_state()
        self.n_rot_ini=self.history_phase_cont[self.i]
        self.reward=0
        

        # Wait 
        return self.state, info


    def render(self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")
    
    def home():
        print('homing...')
        eng.my_quick_home(nargout=0)
        

    def pitch(action_abs):

        print('pitching')
        g.GCommand(f"PAF={action_abs}")


    def read_state(self):
        print('state')
        self.history_time[self.i]=time.time()-self.t_start
        galil_output=c('MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF')
        galil_output[5]-=self.n_rot_ini*360*m[1]['es']
        
        # 
        volts_raw=galil_output[0:5]
        volts=-(volts_raw-self.offset)
        
        # 
        self.history_phase[self.i]= galil_output[5] / m[1]['es'] % 360# In degrees
        self.history_phase_cont[self.i] = galil_output[5] / m[1]['es']  #In degrees
        self.history_volts_raw[self.i]=volts_raw #check formats !!!
        self.history_volts[self.i]=volts #check formats !!!
        ws=10 #window size for fill outliers
        assert(self.i>ws)
        self.filloutliers(ws)
        self.history_pitch_should[self.i]=galil_output[6]/m[2]['ms']
        self.history_pitch_is[self.i]=galil_output[7]/m[2]['es'] #degrees
        self.history_forces_noisy[self.i]=volts[-1]*param['R4'][:,[0,1]] #get Ft and Fr
        b, a = signal.butter(8, 0.01) #to be tuned
        self.history_forces_butter[self.i-ws:self.i+1] = signal.filtfilt(b, a, self.history_forces_noisy[self.i-ws:self.i+1], padlen=0)
        self.history_forces[self.i]=self.history_forces_butter[self.i]
        self.history_forces[self.i,0]-=param['F0'] #remove drag offset 
        Ueff=param['Uinf']*(1+2*param['lambda']*np.cos(self.history_phase[self.i]/360*2*np.pi())+param['lambda']**2)**0.5
        Fsp=param['Csp']*0.5*param['rho']*Ueff**2*param['spr']**2*np.pi*2
        self.history_coeff[self.i,0]=(self.history_forces[self.i,0]*np.cos(np.deg2rad(self.history_pitch_is[self.i]))-self.history_forces[self.i,1]*np.sin(np.deg2rad(self.history_pitch_is[self.i]))+Fsp)/param['f_denom']
        self.history_coeff[self.i,1]=(self.history_forces[self.i,0]*np.sin(np.deg2rad(self.history_pitch_is[self.i]))+self.history_forces[self.i,1]*np.cos(np.deg2rad(self.history_pitch_is[self.i]))+param['Finertial'])/param['f_denom']
        self.i+=1
        self.state=self.history_coeff[self.i]

    def start_E():
        print("Starting motor E")
        g.GCommand('SHE')
        g.GCommand(f"JGE={param['JG']}")
        g.GCommand('BGE')

        #initialize position tracking
        g.GCommand("PTF=1")

    def stop_E():
        print("Stopping motor E")
        g.GCommand("ST")

    def wait_N_rot(self,N):
        print(f"waiting for {N} periods")
        self.read_state()
        phase_ini=self.history_phase_cont[self.i]
        while self.history_phase_cont[self.i]-phase_ini< N*360:
            self.read_state()
            
    def save_data(self,ms=2):
        print("saving data...")
        path=f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms00{ms}mpt{'{:04}'.format(self.episode_counter)}.mat"
        dict={'param':param,'time':self.history_time,'phase':self.history_phase,'phase_cont':self.history_phase_cont,'pitch_is':self.history_pitch_is,'pitch_should':self.history_pitch_should, 'action':self.history_action,'volts_raw':self.history_volts_raw, 'volts': self.history_volts, 'forces_noisy' : self.history_forces_noisy, 'forces_butter': self.history_forces_butter,'forces':self.history_forces,'coeff':self.history_coeff, 'state':self.history_states}
        savemat(path,dict)

        
        
                
        
    def get_offset(self):
        t_offset_pause = 5
        tic=time.time()
        i_start=self.i #should be 0
        while time.time()-tic<t_offset_pause:
            self.history_time[self.i]=time.time()-self.t_start
            self.history_volts_raw[self.i]=c('MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]')
            self.i+=1
        
        self.offset=np.mean(self.history_volts_raw[i_start:self.i+1])
        self.history_volts[i_start:self.i+1]=-(self.history_volts_raw[i_start:self.i+1]-self.offset)
        
    def close(self):
        g.GClose()
        
    
    def filloutliers(self,ws):
        window=self.history_volts[self.i-ws:self.i+1]
        med=np.median(window)
        MAD=1.4826*np.median(np.abs(window-med))
        if np.abs(self.history_volts[self.i]-med)>3/MAD:
            self.history_volts[self.i]=med+3/MAD*np.sign(self.history_volts[self.i]-med)
        if window==np.ones(ws)*med:
            self.history_volts[self.i-ws:self.i+1]=-(self.history_volts_raw[self.i-ws:self.i+1]-self.offset)