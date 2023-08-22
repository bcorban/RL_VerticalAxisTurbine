import numpy as np
from setup_galil import setup_g
# import gymnasium as gym
# from gymnasium.spaces import Box, Discrete
import gym
from gym.spaces import Box
import random
import sys
from typing import Optional
# import torch
import math
import gclib
import matlab.engine
# import time1
from scipy import signal
from scipy.io import savemat
from config_ENV import CONFIG_ENV
from param_matlab import param, m, NI
import load_mat
import getpass
# import ray
# import threading
from multiprocessing import Process, Value, Manager
from ctypes import c_bool
import win_precise_time as wpt

ACTUATE=CONFIG_ENV['ACTUATE']
if ACTUATE:
    Cp_na_=load_mat.load_mat(f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/Cp_phavg.mat")
    Cp_na=np.array(Cp_na_['Cp_phavg']['phavg'])

user = getpass.getuser()
if user == "PIVUSER":
    sys.path.append("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs")

    eng = matlab.engine.start_matlab()
    path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel"
    eng.addpath(path, nargout=0)
    path = "/Users/PIVUSER/Documents/MATLAB"
    eng.addpath(path, nargout=0)
    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")


elif user == "adminit":
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
    def __init__(self) -> None:  # This method is called when creating the environment
        super(CustomEnv, self).__init__()
        # self.dict_env = dict_env
        self.action_space = Box(
            low=CONFIG_ENV["action_lb"], high=CONFIG_ENV["action_hb"], shape=(1,)
        )

        self.observation_space = Box(
            low=np.array([-5, -5, -5]), high=np.array([5, 5, 5])
        )

        self.N_transient_effects = CONFIG_ENV[
            "N_transient_effects"
        ]  # Number of rotations without sampling transitions, at the begining of each episodes
        self.N_max = CONFIG_ENV["N_max"]  # max number of steps in an episode

        self.episode_counter = Value("i", 0)
        print("init")

    def step(
        self, action
    ):  # This method performs one step, i.e. one pitching command, and returns the next state and reward
        
        self.overshoot = False
        if ACTUATE:
            self.action_abs = self.pitch_is.value + action  # in degrees
        else:
            self.action_abs=0

        if self.action_abs > 30:
            self.action_abs = 30
            self.overshoot = True
        elif self.action_abs < -30:
            self.action_abs = -30
            self.overshoot = True

        self.history_action[self.j] = action
        self.history_action_abs[self.j] = self.action_abs
        self.history_timestamp_actions[self.j] = wpt.time() - self.t_start.value

        g.GCommand(f"PAF={int(self.action_abs*m[1]['ms'])}")
       
        if self.nrot.value <= self.N_transient_effects + 3:  # if during transients effects, do not save samples
                info = {"transient": True}
        else:
            info={"transient": False}

        # Check if episode terminated ---------------------------------
        
        if self.nrot.value >= self.N_max:
            print("terminated")
            self.terminated.value = True
            g.GCommand(f"PAF=0")  # pitch back to 0 at the end of an episode
        else:
            self.terminated.value = False

        return np.array(self.state), 0, self.terminated.value, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):  # This method is called at the begining of each episode
        
        if self.episode_counter.value >= 1:  # if not first episode
            print("reset")
            self.process.join()
            self.save_data()  # save data from previous ep

        self.overshoot=False
        self.j = 0  # action counter
        self.episode_counter.value += 1
        self.history_states = np.zeros((100000, 3))
        self.history_action = np.zeros(100000)
        self.history_action_abs = np.zeros(100000)
        self.history_reward = np.zeros(100000)
        self.history_timestamp_actions = np.zeros(100000)
        self.history_phase_actions = np.zeros(100000)
        self.Cp = Value("d", 0)
        self.reading_bool = Value(c_bool, False)
        self.terminated = Value(c_bool, False)
        self.pitch_is = Value("d", 0)
        self.phase=Value("d",0)
        self.nrot = Value("i", 0)
        self.t_start=Value('d',0)
        manager = Manager()
        self.state = manager.list()
        self.state[:] = [0, 0, 0]

        self.process = Process(
            target=continuously_read,
            args=(
                self.terminated,
                self.state,
                self.Cp,
                self.nrot,
                self.pitch_is,
                self.reading_bool,
                self.episode_counter,
                self.t_start,
                self.phase
            ),
        )

        self.process.start()

        info = {}
        # -----------------------------------

        self.reward = 0
        while not self.reading_bool.value:
            pass

        if not return_info:
            return np.array(self.state)

        else:
            return np.array(self.state), info

    def render(self):
        print(f"t={self.j} -- state={self.state} -- reward={self.reward}")

    def save_data(self):  # export data from an episode into a .mat file
        print("saving data...")
        if user == "PIVUSER":
            path = f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms{'{:03}'.format(CONFIG_ENV['ms'])}mpt{'{:03}'.format(self.episode_counter.value)}_2.mat"
            dict = {
                "state": self.history_states[:self.j],
                "action": self.history_action[:self.j],
                "action_abs": self.history_action_abs[:self.j],
                "reward": self.history_reward[:self.j],
                "time_action": self.history_timestamp_actions[:self.j],
                "phase_action": self.history_phase_actions[:self.j],
            }
            savemat(path, dict)

    def close(self):  # close galil connection when closing environment
        print("Closing environment")

        self.terminated.value = True
        self.process.join()
        self.save_data()
        eng.sharx_off(nargout=0)
        g.GClose()

    def get_transition(self):

        next_state, Cp_, phase_ = np.array(self.state), self.Cp.value, int(self.phase.value)
        if not ACTUATE:
            self.reward=0

        else:
            if self.overshoot:
                self.reward = -1
            else:
                # self.reward=(Cp_+0.2)
                # self.reward = max(Cp_,0) / 0.3  # transformation to keep reward roughly between 0 and 1
                self.reward=max(0,(1+(Cp_-Cp_na[phase_]))/2)
                # self.reward=max(0,(1+Cp_)/2)
                # if phase_<180:
                #     self.reward=max(-2,(Cp_-Cp_na[phase_]))
                # else:
                #     self.reward=max(-2,(Cp_-Cp_na[phase_])*5)
                #     if self.action_abs>=5:
                #         self.reward+=1
                # self.reward=max(-2,(Cp_-Cp_na[phase_]))
        self.history_states[self.j] = next_state
        self.history_reward[self.j] = self.reward
        self.history_phase_actions[self.j]=phase_
        self.j += 1

        return np.array([next_state]),np.array([self.reward]) #return state and reward in a vectorized manner for the RL loop
        # --------------------------------------------------------------

def continuously_read(
    terminated, state, Cp, rot_number, pitch_is, reading_bool, episode_counter, t_start, phase_for_reward
):
    N = 1000000
    reading_bool.value = False

    # Initialize all history arrays

    history_phase = np.zeros(N)
    history_phase_cont = np.zeros(N)
    history_pitch_done = np.zeros(N)
    history_pitch_is = np.zeros(N)
    history_dpitch_noisy = np.zeros(N)
    history_dpitch_filtered = np.zeros(N)
    history_volts = np.zeros((N, 5))
    history_volts_raw = np.zeros((N, 5))
    history_time = np.zeros(N)
    history_forces_noisy = np.zeros((N, 3))
    history_forces_noisy_= np.zeros((N, 3))
    history_forces_butter = np.zeros((N, 3))
    history_forces = np.zeros((N, 3))
    history_coeff = np.zeros((N, 2))
    history_Cp = np.zeros(N)
    offset_volts = np.zeros((10000, 5))

    fc_f = 30  # filter parameters
    order_f = 2
    fs = 550
    fc_p = 30
    order_p = 2
    b, a = signal.butter(order_f, fc_f / (fs / 2))  # to be tuned
    b_p, a_p = signal.butter(order_p, fc_p / (fs / 2))
    i = 0  # timestep counter

    g = gclib.py()
    c = g.GCommand
    g.GOpen("192.168.255.200 --direct -s ALL")  # connect galil

    # -------------------------HOMING-----------------------------------------
    print("homing...")
    g.GCommand("OEF=0")

    path = "/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs"
    eng.addpath(path, nargout=0)
    eng.my_quick_home(nargout=0)
    g.GCommand("OEF=2")
    g.GCommand("DEF=0")  # force encoder signal to 0 after homing 
    
    # Reset acceleration and speed after the homing
    setup_g(g)

    # ------------------------------------------------------------------------
    # -------------------------MEASURE OFFSET---------------------------------
    print("Offset")
    t_offset_pause = 5
    tic = wpt.time()
    k = 0  # should be 0
    while wpt.time() - tic < t_offset_pause:
        offset_volts[k] = list(
            map(float, (c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7]")).split())
        )
        k += 1

    offset = np.mean(offset_volts[:k], 0)
    # ------------------------------------------------------------------------

    # ----------------------------START MOTOR AND LOADCELL--------------------

    eng.start_lc(nargout=0)  # Start the loadcell
    t_start.value = wpt.time()
    # t_start.value=eng.workspace['t_start']


    g.GCommand("SHE")
    g.GCommand(f"JGE={int(param['JG'])}")
    g.GCommand("BGE")

    # initialize position tracking
    g.GCommand("SHF")
    g.GCommand("PTF=1")
    
    # --------------------Wait for 3 rotations--------------------------------
    while float(g.GCommand("MG _TPE")) / m[0]["es"] // 360 < 3:
        # time.sleep(0.001)
        pass

    # ------------------------------------------------------------------------
    print("begin reading")

    reading_bool.value = True  # when this bool turns true, the reset() method in the other process can go on
    
    while not terminated.value and i < N:  # while episode not ended and lists not full
        # ----------BEGIN READ STATE-----------------------------------------
        history_time[i] = wpt.time() - t_start.value  # get time

        # read galil output (volts)

        galil_output = np.array(
            list(
                map(
                    float,
                    (c("MG @AN[1],@AN[2],@AN[3],@AN[5],@AN[7],_TPE,_TDF,_TPF")).split(),
                )
            )
        )

        # get voltages for the loads, and substract measured offset
        volts_raw = galil_output[0:5]
        volts = -(volts_raw - offset)
        history_volts_raw[i, :] = volts_raw
        history_volts[i, :] = volts

        # get the phase and pitch of the blade, and the noisy pitch increment
        history_phase[i] = galil_output[5] / m[0]["es"] % 360  # In degrees
        history_phase_cont[i] = galil_output[5] / m[0]["es"]  # In degrees
        rot_number.value = int(galil_output[5] / m[0]["es"] // 360)
        phase_for_reward.value = history_phase[i]
        history_pitch_done[i] = galil_output[6] / m[1]["ms"]  # In degrees
        history_pitch_is[i] = galil_output[7] / m[1]["es"]  # In degrees
        pitch_is.value = history_pitch_is[i]
        if i > 2:
            # history_dpitch_noisy[i] = (history_pitch_is[i] - history_pitch_is[i-1]) / (history_time[i]-history_time[i-1])
            history_dpitch_noisy[i] = np.gradient(history_pitch_is[i-2:i+1], history_time[i-2:i+1],edge_order=2)[-1]
        ws = 100  # window size for filtering
        ws_f = 10 # window size for filloutliers

        if i > ws:
            # ---------------------FILLOUTLIERS-------------------

            window = history_volts[i - ws_f : i + 1]
            med = np.median(window, axis=0)

            MAD = 1.4826 * np.median(
                np.abs(window - np.tile(med, (len(window), 1))), axis=0
            )  # inspired from matlab filloutliers method

            mask = np.abs(history_volts[i] - med) > 2 * MAD

            replace = np.where(
                mask, med + 2 * MAD * np.sign(history_volts[i] - med), history_volts[i]
            )
            # replace = np.where(
            #     mask, med, history_volts[i]
            # )


            history_volts[i] = replace
            
            flatlined = False
            
            for l in range(5): #check if any of the volts channels flatlined
                if np.all(history_volts[i - ws_f : i + 1, l]== med[l] ):
                    history_volts[i - ws_f : i + 1, l] = -(history_volts_raw[i - ws_f : i + 1, l] - offset[l])
   
            # ----------------------------------------------------

            history_forces_noisy[i] = np.dot(
                np.array(history_volts[i]), param["R4"]
            )  # get Fx,Fy and Mz from volts using calibration matrix

            window_= history_forces_noisy[i - ws_f : i + 1]
            med_ = np.median(window_, axis=0)

            # MAD = 1.4826 * np.median(
            MAD_ = 1.4826 * np.median(
                np.abs(window_ - med_), axis=0
            )  # inspired from matlab filloutliers method

            mask_ = np.abs(history_forces_noisy[i] - med_) > 1.5* MAD_

            replace_ = np.where(
                mask_, med_ + 0* MAD_ * np.sign(history_forces_noisy[i] - med_), history_forces_noisy[i]
            )
            history_forces_noisy_[i] = replace_

            


            for l in range(3): #check if any of the volts channels flatlined
                if np.all(history_forces_noisy_[i - ws_f : i + 1, l] == med[l]):

                    history_forces_noisy_[i - ws_f : i + 1, l] = history_forces_noisy[i - ws_f : i + 1, l]
                    # print(f'flatlined {l}',flush=True)
                    # i_list.append(i)



            # filtering step for forces
            history_forces_butter[i - ws : i + 1] = signal.filtfilt(
                b, a, history_forces_noisy_[i - ws : i + 1], axis=0,padlen=0
            )

            # filtering step for dpitch
            history_dpitch_filtered[i - ws : i + 1] = signal.filtfilt(
                b_p, a_p, history_dpitch_noisy[i - ws : i + 1], padlen=0
            )

        else:  # No filtering or filloutliers step
            history_forces_noisy[i] = np.dot(
                np.array(history_volts[i]), param["R4"]
            )  # get Fx,Fy and Mz from volts using calibration matrix

            history_forces_butter[i - ws : i + 1] = history_forces_noisy[i - ws : i + 1]
            history_dpitch_filtered[i - ws : i + 1] = history_dpitch_noisy[i - ws : i + 1] 
        
        history_forces[i, :] = history_forces_butter[i, :]
        history_forces[i, 0] -= param["F0"]  # remove drag offset

        Ueff = (
            param["Uinf"]
            * (
                1
                + 2 * param["lambda"] * np.cos(np.deg2rad(history_phase[i]))
                + param["lambda"] ** 2
            )
            ** 0.5
        )

        Fsp = (  # splitter plate force
            param["Csp"]
            * 0.5
            * param["rho"]
            * Ueff**2
            * param["spr"] ** 2
            * math.pi
            * 2
        )

        # projection and non dimensionalisation of the loads into usable coefficients Ct, Cr
        Ft = (
            history_forces[i, 0] * np.cos(np.deg2rad(history_pitch_is[i]))
            - history_forces[i, 1] * np.sin(np.deg2rad(history_pitch_is[i]))
            + Fsp
        )
        # Compute Cp for reward
        
        Pgen = (Ft*param['R']*param['rotf']*2*math.pi) # generated power

        Pmot = abs(history_forces[i,2]*history_dpitch_filtered[i]) # Mz * omega_F 
        # Pmot=0
        if math.isnan(Pmot):
            print('nan')
            Pmot=0

        Pflow = 0.5*param['rho']*param['Uinf']**3*param['span']*param['R']*2 

        history_Cp[i]=(Pgen-Pmot)/Pflow

        # if i>10:
        #     Cp.value=np.mean(history_Cp[i-9:i+1])
        # else:
        #     Cp.value=history_Cp[i]
        Cp.value=history_Cp[i]
        
        if math.isnan(Cp.value):
            print("Cp",flush=True)
        
        history_coeff[i, 0] = Ft / param["f_denom"]  # compute Ct

        history_coeff[i, 1] = (  # Compute Cr
            history_forces[i, 0] * np.sin(np.deg2rad(history_pitch_is[i]))
            + history_forces[i, 1] * np.cos(np.deg2rad(history_pitch_is[i]))
            + param["Finertial"]
        ) / param["f_denom"]

        # state[:2] = history_coeff[i]  # update state
        state[0] = (history_coeff[i,0] + 0.3)/1.1 #Normalize state (Ct+min_Ct_non_actuated)/max-min
        state[1] = (history_coeff[i,1] + 1.8)/6

        # adding Cr with T/5 time delay to the state. Check in the 500 past measures which one was exactly T/5 seconds away.
        npast = 300
        if i > npast:
            idx = np.searchsorted(
                history_time[i-npast:i], history_time[i] - param["rotT"] / 5, side="left"
            )
            state[2] = (history_coeff[i-npast + idx, 1]+1.8)/6  # add Cr(t-T/5) normalized

        i += 1  # update counter

    print("stop reading", flush=True)
    reading_bool.value = False
    # ----------END READ STATE----------------------------------------

    # ------------STOP MOTORS-----------------------------------------
    g.GCommand("ST")
    g.GClose()
    # ----------------------------------------------------------------

    # ------------SAVE DATA-------------------------------------------
    eng.stop_lc(nargout=0)  # stop loadcell
    if user == "PIVUSER":
        path = f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/ms{'{:03}'.format(CONFIG_ENV['ms'])}mpt{'{:03}'.format(episode_counter.value)}_1.mat"
        dict = {
            "param": param,
            "t_g": history_time[:i-1],
            "phase": history_phase[:i-1],
            "phase_cont": history_phase_cont[:i-1],
            "pitch_is": history_pitch_is[:i-1],
            "pitch_should": history_pitch_done[:i-1],
            "volts_raw_g": history_volts_raw[:i-1],
            "volts_g": history_volts[:i-1],
            "forces_noisy_g": history_forces_noisy[:i-1],
            "forces_butter_g": history_forces_butter[:i-1],
            "forces_g": history_forces[:i-1],
            "force_coeff_g": history_coeff[:i-1],
            "v_offset_g": offset,
            "dpitch_noisy" : history_dpitch_noisy[:i-1],
            "dpitch_filtered" : history_dpitch_filtered[:i-1],
            "Cp" : history_Cp[:i-1]
        }
        savemat(path, dict)
    # ----------------------------------------------------------------
    
    print("Process_ending")
