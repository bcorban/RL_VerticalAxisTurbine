import numpy as np
# import gymnasium as gym
# from gymnasium.spaces import Box, Discrete
import gym
from gym.spaces import Box
import random
import sys
from typing import Optional
import torch
import math
import gclib
import matlab.engine
import time
from scipy import signal
from scipy.io import savemat
from config_ENV import CONFIG_ENV
from param_matlab import param, m, NI
import getpass
import ray
import threading
from multiprocessing import Process, Value,Manager
from ctypes import c_bool

user=getpass.getuser()
if user=='PIVUSER':
    sys.path.append("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs")


    eng = matlab.engine.start_matlab()
    path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
    eng.addpath(path, nargout=0)
    path = "/Users/PIVUSER/Documents/MATLAB"
    eng.addpath(path, nargout=0)
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")


elif user == 'adminit':
    sys.path.append("/home/adminit/RL_VerticalAxisTurbine/Carousel/RL_/envs")


    eng = matlab.engine.start_matlab()
    path = "/home/adminit/RL_VerticalAxisTurbine/Carousel"
    eng.addpath(path, nargout=0)
    path = "/home/adminit/RL_VerticalAxisTurbine"
    eng.addpath(path, nargout=0)
    
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.25 --direct -s ALL")
    print(g.GInfo())


class CustomEnv(gym.Env):
    def __init__(self) -> None: #This method is called when creating the environment
        super(CustomEnv, self).__init__()
        # self.dict_env = dict_env
        self.action_space = Box(
            low=CONFIG_ENV["action_lb"], high=CONFIG_ENV["action_hb"], shape=(1,)
        )

        self.observation_space = Box(low=np.array([-10, -10, -10]), high=np.array([10, 10, 10]))

        self.N_transient_effects = CONFIG_ENV["N_transient_effects"] #Number of rotations without sampling transitions, at the begining of each episodes
        self.N_max = CONFIG_ENV["N_max"] #max number of steps in an episode

        self.episode_counter = Value('i',0)
        print("init")

    def step(self, action):  #This method performs one step, i.e. one pitching command, and returns the next state and reward

        overshoot=False
        self.action_abs=self.pitch_is.value+action #in degrees

        
        if self.action_abs>30:
            self.action_abs=30
            overshoot=True
        elif self.action_abs<-30:
            self.action_abs=-30
            overshoot=True
        

        t_action=time.time()
        
        g.GCommand(f"PAF={int(self.action_abs*m[1]['ms'])}")
        
        next_state=self.state
        next_state=np.array(next_state)

        # Compute reward ---------------------------------------------
        if user=='PIVUSER':
                        
            # if self.nrot.value <= self.N_transient_effects + self.n_rot_ini: #if during transients effects, do not save samples
            #     print("transient")
            #     self.reward = 0
            #     info = {"transient": True}
            # else:
            #     if overshoot:
            #             self.reward=-1 #penalize if going over 30
            #         else:
            #             self.reward = (next_state[1] + 2) / 4  # transformation to keep reward roughly between 0 and 1
            #         info = {"transient": False}
            #         self.history_states[self.j] = next_state 
            #         self.history_action[self.j] = action
            #         self.history_action_abs[self.j]=self.action_abs
            #         self.history_reward[self.j] = self.reward
            #         self.history_timestamp_actions[self.j]=self.history_time[index]
            #         self.j+=1

            if self.j<10: #if during transients effects, do not save samples
                print("transient")
                self.reward = 0
                info = {"transient": True}
                self.j+=1
            else:
                if overshoot:
                    self.reward=-1
                else:
                    self.reward = (next_state[1] + 2) / 4  # transformation to keep reward roughly between 0 and 1
                info = {"transient": False}
                
                self.history_states[self.j] = next_state 
                self.history_action[self.j] = action
                self.history_action_abs[self.j]=self.action_abs
                self.history_reward[self.j] = self.reward
                self.history_timestamp_actions[self.j]=t_action
                self.j+=1
                
                
        elif user == 'adminit':

            if self.j<10: #if during transients effects, do not save samples
                print("transient")
                self.reward = 0
                info = {"transient": True}
                self.j+=1
            else:
                if overshoot:
                    self.reward=-1
                else:
                    self.reward = (next_state[1] + 2) / 4  # transformation to keep reward roughly between 0 and 1
                info = {"transient": False}
                
                self.history_states[self.j] = next_state 
                self.history_action[self.j] = action
                self.history_action_abs[self.j]=self.action_abs
                self.history_reward[self.j] = self.reward
                self.history_timestamp_actions[self.j]=t_action
                self.j+=1

        # --------------------------------------------------------------
        
        # Check if episode terminated ---------------------------------
        if user=='PIVUSER':
        
            if self.nrot.value >= self.N_max: 
                print("terminated")
                self.terminated.value = True
                g.GCommand(f"PAF=0")  # pitch back to 0 at the end of an episode
            else:
                self.terminated.value = False
        elif user == 'adminit':
            if self.j>100: 
                print("terminated")
                self.terminated.value = True
                g.GCommand(f"PAF=0") # pitch back to 0 at the end of an episode
            else:
                self.terminated.value = False
        truncated = False
        # print(f"\n one step takes {time.time()-t_1}s \n")
        # Return step information

        return next_state, self.reward, self.terminated.value, info

    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None): #This method is called at the begining of each episode
        self.episode_counter.value += 1
        
        self.j = 0 #action counter
        
        if self.episode_counter.value > 1:  # if not first episode
            print("reset")
            if user =='PIVUSER':
                eng.stop_lc(nargout=0) #stop loadcell
            
            self.process.join()
            self.save_data() #save data from previous ep
            
            self.history_states=np.zeros((100000,3))
            self.history_action=np.zeros(100000)
            self.history_action_abs=np.zeros(100000)
            self.history_reward=np.zeros(100000)
            self.history_timestamp_actions=np.zeros(100000)
            
            self.reading_bool=Value(c_bool,False)
            self.terminated=Value(c_bool,False)
            self.pitch_is=Value('d',0)
            self.nrot=Value('i',0)
            manager=Manager()
            self.state=manager.list()
            self.state[:]=[0,0,0]
            
            self.process=Process(target=continuously_read,args=(self.terminated,self.state,self.nrot,self.pitch_is,self.reading_bool,self.episode_counter))
            
            if user =='PIVUSER':
                try:
                    eng.start_lc(nargout=0)  # Start the loadcell
                except:
                    print("Did not start loadcell !!")
            
            self.process.start()
            
                
        else: # first episode : start load cell, get offset, start motor
            print("first episode")
            
            self.history_states=np.zeros((100000,3))
            self.history_action=np.zeros(100000)
            self.history_action_abs=np.zeros(100000)
            self.history_reward=np.zeros(100000)
            self.history_timestamp_actions=np.zeros(100000)
            self.reading_bool=Value(c_bool,False)
            self.terminated=Value(c_bool,False)
            self.pitch_is=Value('d',0)
            self.nrot=Value('i',0)
            manager=Manager()
            self.state=manager.list()
            self.state[:]=[0,0,0]
            self.process=Process(target=continuously_read,args=(self.terminated,self.state,self.nrot,self.pitch_is,self.reading_bool,self.episode_counter))

            if user =='PIVUSER':
                try:
                        eng.start_lc(nargout=0)  # Start the loadcell
                except:
                    print("Did not start loadcell !!")
            
            self.process.start()
           
            
        info = {}
        # -----------------------------------
        
        self.reward = 0
        while not self.reading_bool.value:

            pass

        if not return_info:

            return np.array(self.state)
            
        else:
            return np.array(self.state),info

        
    
    def render(self):
        print(f"t={self.j} -- state={self.state} -- reward={self.reward}")

         

        
            
            
    def save_data(self): #export data from an episode into a .mat file
        print("saving data...")
        if user=="PIVUSER":
            path=f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms001mpt{'{:03}'.format(self.episode_counter.value)}_2.mat"
            dict={
                  'state':self.history_states,
                  'action':self.history_action,
                  'action_abs':self.history_action_abs,
                  'reward':self.history_reward,
                  'time_action':self.history_timestamp_actions}
            savemat(path,dict)

            
    def close(self): #close galil connection when closing environment
        print("Closing environment")
        self.terminated=True
        if user =='PIVUSER':
            eng.stop_lc(nargout=0) #stop loadcell
        self.process.join()
        g.GClose()
        self.save_data()
        
def continuously_read(terminated,state,rot_number,pitch_is,reading_bool,episode_counter):
    
    N = 1000000
    reading_bool.value=False
    #Initialize all history arrays
    
    history_phase = np.zeros(N)
    history_phase_cont = np.zeros(N)
    history_pitch_should = np.zeros(N)
    history_pitch_is = np.zeros(N)
    history_volts = np.zeros((N, 5))
    history_volts_raw = np.zeros((N, 5))
    history_time = np.zeros(N)
    history_forces_noisy = np.zeros((N, 2))
    history_forces_butter = np.zeros((N, 2))
    history_forces = np.zeros((N, 2))
    history_coeff = np.zeros((N, 2))
    offset_volts=np.zeros((10000,5))
    i = 0 #timestep counter 
    
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL") #connect galil

    #-------------------------HOMING-----------------------------------------
    print("homing...")    
    g.GCommand("OEF=0")
    # eng.my_quick_home(nargout=0)
    g.GCommand("OEF=2")
    #------------------------------------------------------------------------
    #-------------------------MEASURE OFFSET---------------------------------
    t_offset_pause = 5
    tic = time.time()
    k=0  # should be 0
    while time.time() - tic < t_offset_pause:
        offset_volts[k]= list(
            map(float, (c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]")).split())
        )
        k += 1
        

    offset = np.mean(history_volts_raw[:k])
    #------------------------------------------------------------------------
    
    #-------START MOTOR------------------------------------------------------
    t_start=time.time()
    g.GCommand("SHE")
    g.GCommand(f"JGE={int(param['JG'])}")
    g.GCommand("BGE")

    #initialize position tracking
    g.GCommand("SHF")
    g.GCommand("PTF=1")
    #--------------------Wait for 3 rotations--------------------------------
    # while float(g.GCommand("MG _TPE"))/m[0]["es"] // 360 <3:
    #     time.sleep(0.001)
    
    #------------------------------------------------------------------------
    print("begin reading")
    reading_bool.value=True
    t=time.time()
    while not terminated.value and i<N:
        # if i%1000==0 and i>1:
        #     print(f"Reading freq : {1000/(time.time()-t)}")
        #     t=time.time()

        #----------BEGIN READ STATE-----------------------------------------
        
        history_time[i] = time.time() - t_start #get time

        #read galil output (volts)
        
        galil_output = list(map(float,(c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF")).split()))
     
        # get voltages for the loads, and substract measured offset
        volts_raw = galil_output[0:5] 
        volts = -(volts_raw - offset)
        history_volts_raw[i] = volts_raw  
        history_volts[i] = volts 

        #get the phase and pitch of the blade
        history_phase[i] = galil_output[5] /m[0]["es"] % 360  # In degrees
        history_phase_cont[i] = galil_output[5] /m[0]["es"]  # In degrees
        rot_number.value=int(galil_output[5] /m[0]["es"] // 360) 
        history_pitch_should[i] = galil_output[6] /m[1]["ms"] # In degrees
        history_pitch_is[i] = galil_output[7] /m[1]["es"]  # In degrees
        pitch_is.value=history_pitch_is[i]
        ws = 10  # window size for fill outliers and filtering

        if i > ws:
            
        
            #---------------------FILLOUTLIERS-------------------
            window =np.array(history_volts[i - ws : i + 1])
            med = np.median(window,axis=0
                            )
            MAD = 1.4826 * np.median(np.abs(window - np.tile(med,(len(window),1))),axis=0)

            mask= np.abs(history_volts[i] - med) >3* MAD

            replace=np.where(mask, med + 3 * MAD * np.sign(
                    history_volts[i] - med
                ),history_volts[i])
            
            history_volts[i]=replace
            flatlined=False
            for l in range(5):
                if np.all(window[:,l]==med[l]*np.ones(len(window))):
                    flatlined=True
                    break
            if flatlined:
                # print('flatlined')
                history_volts[i - ws : i + 1,:] = -(
                    history_volts_raw[i - ws : i + 1,:] - offset
                )
            #----------------------------------------------------
            
            
            history_forces_noisy[i] = (
                np.dot(np.array(volts),param["R4"][:, 0:2])
            )  # get Fx and Fy from volts using calibration matrix

            #filtering step
            b, a = signal.butter(8, 0.01)  # to be tuned
            history_forces_butter[i - ws : i + 1] = signal.filtfilt(
                b, a, history_forces_noisy[i - ws : i + 1], padlen=0
            )
            
        else:
            
            history_forces_noisy[i] = (
                np.dot(np.array(volts),param["R4"][:, 0:2])
            )  # get Fx and Fy from volts using calibration matrix

            #No filtering step
            history_forces_butter[i - ws : i + 1] = history_forces_noisy[i - ws : i + 1]

        history_forces[i] = history_forces_butter[i]
        history_forces[i, 0] -= param["F0"]  # remove drag offset
        
        Ueff = ( 
            param["Uinf"]
            * (
                1
                + 2
                * param["lambda"]
                * np.cos(np.deg2rad(history_phase[i]))
                + param["lambda"] ** 2
            )
            ** 0.5
        )
        
        Fsp = ( #splitter plate force
            param["Csp"]
            * 0.5
            * param["rho"]
            * Ueff**2
            * param["spr"] ** 2
            * np.pi
            * 2
        )
        
        #projection and non dimensionalisation of the loads into usable coefficients Ct, Cr
        history_coeff[i, 0] = ( #Ct
            history_forces[i, 0]
            * np.cos(np.deg2rad(history_pitch_is[i]))
            - history_forces[i, 1]
            * np.sin(np.deg2rad(history_pitch_is[i]))
            + Fsp
        ) / param["f_denom"] 
        
        history_coeff[i, 1] = ( #Cr
            history_forces[i, 0]
            * np.sin(np.deg2rad(history_pitch_is[i]))
            + history_forces[i, 1]
            * np.cos(np.deg2rad(history_pitch_is[i]))
            + param["Finertial"]
        ) / param["f_denom"]
        
        state[:2] = history_coeff[i] #update state
        
        #adding Cr with T/5 time delay. Check in the 500 past measures which one was exactly T/5 seconds away.
        npast = 500
        if i > npast:

            idx=np.searchsorted(history_time[-npast:],history_time[i]-param["rotT"]/5,side="left")
            state[2] = history_coeff[-npast+idx, 1] #add Cr(t-T/5)

        i += 1 #update counter   
        # print(i,flush=True)
    print("stop reading",flush=True)
    reading_bool.value=False
    #----------END READ STATE----------------------------------------
    
    #------------STOP MOTORS-----------------------------------------
    g.GCommand("ST")
    g.GClose()
    #----------------------------------------------------------------
    
    #------------SAVE DATA-------------------------------------------
    if user=="PIVUSER":
        path=f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms001mpt{'{:03}'.format(episode_counter.value)}_1.mat"
        dict={'param':param,
              'time':history_time,
              'phase':history_phase,
              'phase_cont':history_phase_cont,
              'pitch_is':history_pitch_is,
              'pitch_should':history_pitch_should, 
              'volts_raw':history_volts_raw, 
              'volts': history_volts, 
              'forces_noisy' : history_forces_noisy, 
              'forces_butter': history_forces_butter,
              'forces':history_forces,
              'coeff':history_coeff, 
              }
        savemat(path,dict)
    #----------------------------------------------------------------
    

    print("Process_ending")

