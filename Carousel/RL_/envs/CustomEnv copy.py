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
from multiprocessing import Process

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
    print(g.GInfo())

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

        self.N_ep_without_homing = CONFIG_ENV["N_ep_without_homing"] #number of episodes between each homing procedures
        self.N_transient_effects = CONFIG_ENV["N_transient_effects"] #Number of rotations without sampling transitions, at the begining of each episodes
        self.N_max = CONFIG_ENV["N_max"] #max number of steps in an episode

        self.episode_counter = 0
        print("init")

    def step(self, action):  #This method performs one step, i.e. one pitching command, and returns the next state and reward
        overshoot=False
        self.action_abs=self.history_pitch_is[self.i]+action #in degrees

        
        if self.action_abs>30:
            self.action_abs=30
            overshoot=True
        elif self.action_abs<-30:
            self.action_abs=-30
            overshoot=True
        


        self.action_flag=True

        while self.action_flag:
            pass
        # print(time.time()-self.t1)

        index,next_state=self.i,self.state
        
        nrot = (
            self.history_phase_cont[index] // 360
        )  #current number of rotation since begining of the episode
        
        # Compute reward ---------------------------------------------
        if user=='PIVUSER':
                        
            # if nrot <= self.N_transient_effects + self.n_rot_ini: #if during transients effects, do not save samples
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
                self.history_timestamp_actions[self.j]=self.history_time[index]
                self.j+=1
                
                
        # elif user == 'adminit':

        #     if self.j<10: #if during transients effects, do not save samples
        #         print("transient")
        #         self.reward = 0
        #         info = {"transient": True}
        #         self.j+=1
        #     else:
        #         if overshoot:
        #             self.reward=-1
        #         else:
        #             self.reward = (next_state[1] + 2) / 4  # transformation to keep reward roughly between 0 and 1
        #         info = {"transient": False}
        #         self.history_states[self.j] = next_state 
        #         self.history_action[self.j] = action
        #         self.history_action_abs[self.j]=self.action_abs
        #         self.history_reward[self.j] = self.reward
        #         self.history_timestamp_actions[self.j]=self.history_time[index]
        #         self.j+=1
        #--------------------------------------------------------------
        
        # Check if episode terminated ---------------------------------
        if user=='PIVUSER':
        
            if nrot >= self.N_max: 
                print("terminated")
                self.terminated = True
                self.pitch(0)  # pitch back to 0 at the end of an episode
            else:
                self.terminated = False
        elif user == 'adminit':
            if self.j>100: 
                print("terminated")
                self.terminated = True
                self.pitch(0)  # pitch back to 0 at the end of an episode
            else:
                self.terminated = False
        truncated = False
        # print(f"\n one step takes {time.time()-t_1}s \n")
        # Return step information
        
        return next_state, self.reward, self.terminated, info

    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None): #This method is called at the begining of each episode
        self.episode_counter += 1
        
        # N = 100000
        # #Initialize all history arrays
        # self.state=np.zeros(3)
        # self.history_phase = np.zeros(N)
        # self.history_phase_cont = np.zeros(N)
        # self.history_pitch_should = np.zeros(N)
        # self.history_pitch_is = np.zeros(N)
        # self.history_states = np.zeros((N, 3))
        # self.history_volts = np.zeros((N, 5))
        # self.history_volts_raw = np.zeros((N, 5))
        # self.history_time = np.zeros(N)
        # self.history_forces_noisy = np.zeros((N, 2))
        # self.history_forces_butter = np.zeros((N, 2))
        # self.history_forces = np.zeros((N, 2))
        # self.history_coeff = np.zeros((N, 2))
        # self.history_action = np.zeros(N)
        # self.history_action_abs = np.zeros(N)
        # self.history_reward = np.zeros(N)
        # self.history_timestamp_actions=np.zeros(N)
        # self.i = 0 #timestep counter for continuous_read
        self.j = 0 #action counter
        self.action_flag=False
        self.terminated=False
        self.daemon=threading.Thread(target=self.continuously_read,daemon=True,name="state_reader")
    
        # print(self.episode_counter)
        if self.episode_counter > 1:  # if not first episode
            print("reset")
            if user =='PIVUSER':
                eng.stop_lc(nargout=0) #stop loadcell
            self.daemon.join()
            print("joined")
            self.save_data() #save data from previous ep
            if self.episode_counter % self.N_ep_without_homing == 0:  # if homing needed
                print("homing")
                self.stop_E()
                self.home()
                if user =='PIVUSER':
                    try:
                        eng.start_lc(nargout=0)  # Start the loadcell
                    except:
                        print("Did not start loadcell !!")
                self.t_start = time.time()
                
                self.get_offset() #get new offset while E stopped
                self.start_E()
                self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
                self.daemon.start()
                self.wait_N_rot(3) #wait a few rotations 
                
            else:
                if user =='PIVUSER':
                    try:
                        eng.start_lc(nargout=0)  # Start the loadcell
                    except:
                        print("Did not start loadcell !!")
                self.t_start = time.time()
                self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
                self.daemon.start()
        else: # first episode : start load cell, get offset, start motor
            print("first episode")
            if user =='PIVUSER':
                try:
                        eng.start_lc(nargout=0)  # Start the loadcell
                except:
                    print("Did not start loadcell !!")
            self.t_start = time.time()
            self.get_offset()
            self.start_E()
            self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
            self.daemon.start()
            self.wait_N_rot(3)
            
        info = {}
        # -----------------------------------
        
        # self.read_state()

        self.reward = 0
        # if user=='adminit':
        self.i_begining=self.i
        
        if not return_info:
            return self.state
        else:
            return self.state,info

        
    
    def render(self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")

    def home(self): #calls homing procedure from matlab
        print("homing...")
        # eng.my_quick_home(nargout=0)

    def read_state(self): #reads current state of the blade : from voltages to forces and position 
        # print("reading state")
        self.history_time[self.i] = time.time() - self.t_start #get time

        #read galil output (volts)
        
        galil_output = list(map(float,(c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF")).split()))
           

        galil_output[5]-=self.n_rot_ini*360*m[0]['es'] #substract the starting phase of the episode

        # get voltages for the loads, and substract measured offset
        volts_raw = galil_output[0:5] 
        volts = -(volts_raw - self.offset)
        self.history_volts_raw[self.i] = volts_raw  
        self.history_volts[self.i] = volts 

        #get the phase and pitch of the blade
        self.history_phase[self.i] = galil_output[5] /m[0]["es"] % 360  # In degrees
        self.history_phase_cont[self.i] = galil_output[5] /m[0]["es"]  # In degrees
        self.history_pitch_should[self.i] = galil_output[6] /m[1]["ms"] # In degrees
        self.history_pitch_is[self.i] = galil_output[7] /m[1]["es"]  # In degrees
         
        ws = 10  # window size for fill outliers and filtering

        if self.i > ws:
            self.filloutliers(ws)
            
            if user=='PIVUSER':
                self.history_forces_noisy[self.i] = (
                    np.dot(np.array(volts),param["R4"][:, 0:2])
                )  # get Fx and Fy from volts using calibration matrix
            elif user=='adminit':
                self.history_forces_noisy[self.i] = (
                    volts[:2]
                )  # DUMMY LINE TO DELETE
                
            #filtering step
            b, a = signal.butter(8, 0.01)  # to be tuned
            self.history_forces_butter[self.i - ws : self.i + 1] = signal.filtfilt(
                b, a, self.history_forces_noisy[self.i - ws : self.i + 1], padlen=0
            )
        else:
           
            
            if user=='PIVUSER':
                self.history_forces_noisy[self.i] = (
                    np.dot(np.array(volts),param["R4"][:, 0:2])
                )  # get Fx and Fy from volts using calibration matrix
            elif user=='adminit':
                self.history_forces_noisy[self.i] = (
                    volts[:2]
                )  # DUMMY LINE TO DELETE
                
            #No filtering step
            
            self.history_forces_butter[self.i - ws : self.i + 1] = self.history_forces_noisy[self.i - ws : self.i + 1]
        self.history_forces[self.i] = self.history_forces_butter[self.i]
        self.history_forces[self.i, 0] -= param["F0"]  # remove drag offset
        
        Ueff = ( 
            param["Uinf"]
            * (
                1
                + 2
                * param["lambda"]
                * np.cos(np.deg2rad(self.history_phase[self.i]))
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
        self.history_coeff[self.i, 0] = ( #Ct
            self.history_forces[self.i, 0]
            * np.cos(np.deg2rad(self.history_pitch_is[self.i]))
            - self.history_forces[self.i, 1]
            * np.sin(np.deg2rad(self.history_pitch_is[self.i]))
            + Fsp
        ) / param["f_denom"] 
        
        self.history_coeff[self.i, 1] = (#Cr
            self.history_forces[self.i, 0]
            * np.sin(np.deg2rad(self.history_pitch_is[self.i]))
            + self.history_forces[self.i, 1]
            * np.cos(np.deg2rad(self.history_pitch_is[self.i]))
            + param["Finertial"]
        ) / param["f_denom"]
        
        self.state[:2] = self.history_coeff[self.i] #update state
        
        #adding Cr with T/5 time delay. Check in the 500 past measures which one was exactly T/5 seconds away.
        npast = 500
        if self.i > npast:
            t_2=time.time()
            idx=np.searchsorted(self.history_time[-npast:],self.history_time[self.i]-param["rotT"]/5,side="left")
            # print(f"searchsort takes {time.time()-t_2}s")
            self.state[2] = self.history_coeff[-npast+idx, 1] #add Cr(t-T/5)

        self.i += 1 #update counter
        
        # print(f"\n reading state takes {time.time()-self.history_time[self.i-1]- self.t_start}s \n")

    def start_E(self): #Start motor E
        print("Starting motor E")
        # g.GCommand("SHE")
        # g.GCommand(f"JGE={int(param['JG'])}")
        # g.GCommand("BGE")

        # # initialize position tracking
        # g.GCommand("SHF")
        # g.GCommand("PTF=1")

    def stop_E(self): #Stop motor E
        print("Stopping motor E")
        # if user=="PIVUSER":
            # g.GCommand("ST")
        
    def pitch(self,action_abs): #perform an absolute pitching order
        # print("pitching")
        # if user=="PIVUSER":
        # t=time.time()
        g.GCommand(f"PAF={int(action_abs*m[1]['ms'])}")
        # print(time.time()-t)
            # time.sleep(0.001)
        # return

    def wait_N_rot(self, N): #Wait N rotations from motor E
        print(f"waiting for {N} periods")
        phase_ini = self.history_phase_cont[self.i]
        
        i_ini= self.i
        
        # if user=="PIVUSER":
        #     while self.history_phase_cont[self.i] - phase_ini < N * 360:
        #         time.sleep(0.001)
        #         
        # elif user=="adminit":
        while self.i-i_ini<15:
            time.sleep(0.001)
                
        
            
            
    def save_data(self, ms=2): #export data from an episode into a .mat file
        print("saving data...")
        # if user=="PIVUSER":
            # path=f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms00{ms}mpt{'{:04}'.format(self.episode_counter)}.mat"
            # dict={'param':param,
            #       'time':self.history_time,
            #       'phase':self.history_phase,
            #       'phase_cont':self.history_phase_cont,
            #       'pitch_is':self.history_pitch_is,
            #       'pitch_should':self.history_pitch_should, 
            #       'volts_raw':self.history_volts_raw, 
            #       'volts': self.history_volts, 
            #       'forces_noisy' : self.history_forces_noisy, 
            #       'forces_butter': self.history_forces_butter,
            #       'forces':self.history_forces,
            #       'coeff':self.history_coeff, 
            #       'state':self.history_states,
            #       'action':self.history_action,
            #       'action_abs':self.history_action_abs,
            #       'reward':self.history_reward,
            #       'time_action':self.history_timestamp_actions}
            # savemat(path,dict)

    def get_offset(self): #measure offset 
        t_offset_pause = 5
        tic = time.time()
        i_start = self.i  # should be 0
        while time.time() - tic < t_offset_pause:
            self.history_time[self.i] = time.time() - self.t_start
            self.history_volts_raw[self.i] = list(
                map(float, (c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]")).split())
            )
            self.i += 1
            # print(self.i)

        self.offset = np.mean(self.history_volts_raw[i_start : self.i + 1])
        self.history_volts[i_start : self.i + 1] = -(
            self.history_volts_raw[i_start : self.i + 1] - self.offset
        )


    def filloutliers(self, ws): #replace outlier values if they exceed three times the MAD value around the median, in a window of ws timesteps
        window =np.array(self.history_volts[self.i - ws : self.i + 1])
        med = np.median(window,axis=0
                        )
        MAD = 1.4826 * np.median(np.abs(window - np.tile(med,(len(window),1))),axis=0)

        mask= np.abs(self.history_volts[self.i] - med) >3* MAD

        replace=np.where(mask, med + 3 * MAD * np.sign(
                self.history_volts[self.i] - med
            ),self.history_volts[self.i])
        
        self.history_volts[self.i]=replace
        flatlined=False
        for l in range(5):
            if np.all(window[:,l]==med[l]*np.ones(len(window))):
                flatlined=True
                break
        if flatlined:
            # print('flatlined')
            self.history_volts[self.i - ws : self.i + 1,:] = -(
                self.history_volts_raw[self.i - ws : self.i + 1,:] - self.offset
            )
            
            
    def close(self): #close galil connection when closing environment
        print("Closing environment")
        self.terminated=True
        if user =='PIVUSER':
            eng.stop_lc(nargout=0) #stop loadcell
        self.daemon.join()
        self.save_data()
        self.stop_E()
        g.GClose()
        
def continuously_read(terminated,state_communication,offset):
    N = 100000
    #Initialize all history arrays
    state=np.zeros(3)
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
    i = 0 #timestep counter for continuous_read
    
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")
    t_start=time.time()
    n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360

    while not terminated:
      
        history_time[i] = time.time() - t_start #get time

        #read galil output (volts)
        
        galil_output = list(map(float,(c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF")).split()))
           

        galil_output[5]-=n_rot_ini*360*m[0]['es'] #substract the starting phase of the episode

        # get voltages for the loads, and substract measured offset
        volts_raw = galil_output[0:5] 
        volts = -(volts_raw - offset)
        history_volts_raw[i] = volts_raw  
        history_volts[i] = volts 

        #get the phase and pitch of the blade
        history_phase[i] = galil_output[5] /m[0]["es"] % 360  # In degrees
        history_phase_cont[i] = galil_output[5] /m[0]["es"]  # In degrees
        history_pitch_should[i] = galil_output[6] /m[1]["ms"] # In degrees
        history_pitch_is[i] = galil_output[7] /m[1]["es"]  # In degrees
         
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








    print("Process_ending")

