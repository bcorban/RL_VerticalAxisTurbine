import numpy as np
# import gymnasium as gym
# from gymnasium.spaces import Box, Discrete
import gym
from gym.spaces import Box
import random
import sys
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


user=getpass.getuser()
if user=='PIVUSER':
    sys.path.append("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs")


    eng = matlab.engine.staCustomEnvrt_matlab()
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
        self.observation_space = Box(low=np.array([-10, -10]), high=np.array([10, 10]))

        self.N_ep_without_homing = CONFIG_ENV["N_ep_without_homing"] #number of episodes between each homing procedures
        self.N_transient_effects = CONFIG_ENV["N_transient_effects"] #Number of rotations without sampling transitions, at the begining of each episodes
        self.N_max = CONFIG_ENV["N_max"] #max number of steps in an episode

        self.episode_counter = 0
        print("init")

    def step(self, action):  #This method performs one step, i.e. one pitching command, and returns the next state and reward
        print("step")
        
        self.pitch(action) #perform the pitching action

        self.read_state() #update state

        

        nrot = (
            self.history_phase_cont[self.i] // 360
        )  #current number of rotation since begining of the episode

        # Compute reward ---------------------------------------------
        if nrot <= self.N_transient_effects + self.n_rot_ini: #if during transients effects, do not save samples
            print("transient")
            self.reward = 0
            info = {"transient": True}
        else:
            self.reward = (self.state[1] + 2) / 4  # transformation to keep reward roughly between 0 and 1
            info = {"transient": False}
            self.history_states[self.i] = self.state 
            self.history_action[self.i] = action
            self.history_reward[self.i] = self.reward
            
        #--------------------------------------------------------------
        
        # Check if episode terminated ---------------------------------
        if nrot >= self.N_max: 
            print("terminated")
            terminated = True
            self.pitch(0)  # pitch back to 0 at the end of an episode
        else:
            terminated = False

        truncated = False

        # Return step information
        return self.state, self.reward, terminated,  info

    def reset(self, *, seed=None, options=None): #This method is called at the begining of each episode
        self.episode_counter += 1
        
        N = 100000
        #Initialize all history arrays
        self.history_phase = np.zeros(N)
        self.history_phase_cont = np.zeros(N)
        self.history_pitch_should = np.zeros(N)
        self.history_pitch_is = np.zeros(N)
        self.history_states = np.zeros((N, 4))
        self.history_volts = np.zeros((N, 5))
        self.history_volts_raw = np.zeros((N, 5))
        self.history_time = np.zeros(N)
        self.history_forces_noisy = np.zeros((N, 2))
        self.history_forces_butter = np.zeros((N, 2))
        self.history_forces = np.zeros((N, 2))
        self.history_coeff = np.zeros((N, 2))
        self.history_action = np.zeros(N)
        self.history_reward = np.zeros(N)
        self.i = 0 #iteration/timestep counter
        
        print(self.episode_counter)
        if self.episode_counter > 1:  # if not first episode
            print("not first ep")
            if user =='PIVUSER':
                eng.stop_lc(nargout=0) #stop loadcell
            self.save_data() #save data from previous ep

            if self.episode_counter % self.N_ep_without_homing == 0:  # if homing needed
                print("homing")
                self.stop_E()
                self.home()
                if user =='PIVUSER':
                    eng.start_lc(nargout=0)  # Start the loadcell
                self.t_start = time.time()
                
                self.get_offset() #get new offset while E stopped
                self.start_E()
                self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
                self.wait_N_rot(3) #wait a few rotations 

            else:
                if user =='PIVUSER':
                    eng.start_lc(nargout=0)  # Start the loadcell
                self.t_start = time.time()
                self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
                
        else: # first episode : start load cell, get offset, start motor
            print("first episode")
            if user =='PIVUSER':
                eng.start_lc(nargout=0)  # Start the loadcell
            self.t_start = time.time()
            self.get_offset()
            self.start_E()
            self.n_rot_ini=float(c("MG _TPE"))/m[0]["es"] // 360 #get initial number of rotation since galil start
            self.wait_N_rot(3)
            
        info = {}
        # -----------------------------------
        
        self.read_state()

        self.reward = 0

        # Wait
        return self.state
    def render(self):
        print(f"t={self.t} -- state={self.state} -- reward={self.reward}")

    def home(self): #calls homing procedure from matlab
        print("homing...")
        eng.my_quick_home(nargout=0)

    def read_state(self): #reads current state of the blade : from voltages to forces and position 
        print("state")
        self.history_time[self.i] = time.time() - self.t_start #get time

        #read galil output (volts)
        galil_output = list(map(float,(c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF")).split()))
        # print(f"\n {galil_output} \n")
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
        assert self.i > ws
        self.filloutliers(ws)
        
        if user=='PIVUSER':
            self.history_forces_noisy[self.i] = (
                volts[-1] * param["R4"][:, [0, 1]]
            )  # get Fx and Fy from volts using calibration matrix
        elif user=='adminit':
            self.history_forces_noisy[self.i] = (
                volts[-1]
            )  # DUMMY LINE TO DELETE
            
        #filtering step
        b, a = signal.butter(8, 0.01)  # to be tuned
        self.history_forces_butter[self.i - ws : self.i + 1] = signal.filtfilt(
            b, a, self.history_forces_noisy[self.i - ws : self.i + 1], padlen=0
        )
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
        
        self.i += 1 #update counter
        self.state = self.history_coeff[self.i] #update state

    def start_E(self): #Start motor E
        print("Starting motor E")
        # g.GCommand("SHE")
        # g.GCommand(f"JGE={int(param['JG'])}")
        # g.GCommand("BGE")

        # # initialize position tracking
        # g.GCommand("PTF=1")

    def stop_E(self): #Stop motor E
        print("Stopping motor E")
        # g.GCommand("ST")
        
    def pitch(self,action_abs): #perform an absolute pitching order
        print("pitching")
        # g.GCommand(f"PAF={action_abs}")
        
    def wait_N_rot(self, N): #Wait N rotations from motor E
        print(f"waiting for {N} periods")
        # self.read_state()
        # phase_ini = self.history_phase_cont[self.i]
        # while self.history_phase_cont[self.i] - phase_ini < N * 360:
        #     self.read_state()
        self.read_state()
        i_ini= self.i
        while self.i-i_ini<15: #DUMMY LINE TO DELETE
            self.read_state()

    def save_data(self, ms=2): #export data from an episode into a .mat file
        print("saving data...")
        # path=f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms00{ms}mpt{'{:04}'.format(self.episode_counter)}.mat"
        # dict={'param':param,'time':self.history_time,'phase':self.history_phase,'phase_cont':self.history_phase_cont,'pitch_is':self.history_pitch_is,'pitch_should':self.history_pitch_should, 'action':self.history_action,'volts_raw':self.history_volts_raw, 'volts': self.history_volts, 'forces_noisy' : self.history_forces_noisy, 'forces_butter': self.history_forces_butter,'forces':self.history_forces,'coeff':self.history_coeff, 'state':self.history_states}
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
        window = self.history_volts[self.i - ws : self.i + 1]
        med = np.median(window)
        MAD = 1.4826 * np.median(np.abs(window - med))

        mask= np.abs(self.history_volts[self.i] - med) > MAD

        replace=np.where(mask, med + 3 / MAD * np.sign(
                self.history_volts[self.i] - med
            ),self.history_volts[self.i])
        self.history_volts[self.i]=replace
        # if np.abs(self.history_volts[self.i] - med) > 3 / MAD:
        #     self.history_volts[self.i] = med + 3 / MAD * np.sign(
        #         self.history_volts[self.i] - med
        #     )
        if window == np.ones((ws,5)) * med:
            print('flatlined')
            self.history_volts[self.i - ws : self.i + 1,:] = -(
                self.history_volts_raw[self.i - ws : self.i + 1,:] - self.offset
            )
            
    def close(self): #close galil connection when closing environment
        print("Closing environment")
        if user =='PIVUSER':
            eng.stop_lc(nargout=0) #stop loadcell
        self.save_data()
        self.stop_E()
        g.GClose()
    

