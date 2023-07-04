# import torch
import numpy as np
# import time
# import threading
# import RL_ 
# import gym
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
# import gclib
# from multiprocessing import Process, Value,Manager
import ctypes
from load_mat import load_mat
from param_matlab import param, m, NI
import math
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy import signal
# fc_f = 20  # filter parameters
# order_f = 4
# fs = 550
# fc_p = 30
# order_p = 2
# b, a = signal.butter(order_f, fc_f / (fs / 2)) 
# print(a,b)
# MS = 213.3333
# ES = 666.6666


# def f (N,t_start,t_a):
#     print("process starts")
#     g = gclib.py()
#     c = g.GCommand
#     g.GOpen("192.168.255.200 --direct -s ALL")
#     tpf=np.zeros(N)
#     t=np.zeros(N)
#     t_start.value=time.time()
#     for i in range(N):
#         # print(i)
        
#         tpf[i]=float(c("MG _TPF"))/(ES)
#         t[i]=time.time()-t_start.value
#         time.sleep(0.0001)
#     np.savez('test.npz',
#              tpf=tpf,
#              t=t)
#     g.GClose()
    
# if __name__ == '__main__':
#     g = gclib.py()
#     c = g.GCommand
#     g.GOpen("192.168.255.200 --direct -s ALL")

#     ACF = 2
#     SPF = 1000000
#     KSF = 0.5  # Between 0.25 (low smoothing) and 64 (high smoothing)

#     c(f"ACF={round(256000 * ACF)}")
#     c(f"DCF={round(256000 * ACF)}")
#     c(f"SPF={SPF}")  # Default value: 10666
#     c(f"KSF={KSF}")  # Default value: 10666

#     g.GCommand("SHF")
#     g.GCommand("PTF=1")
#     g.GCommand("OEF=2")
#     t_start=Value('d',0)
#     t_a=Value('d',0)
#     p=Process(target=f, args=(500,t_start,t_a))
#     p.start()
#     while p.is_alive():
#         if t_start.value>0 and time.time()-t_start.value>1:
#             c(f"PAF={int(6*MS)}")
#             t_a.value=time.time()-t_start.value
#             break
    
#     p.join()
#     g.GCommand("PAF=0")
#     g.GCommand("OEF=0")
    
#     g.GClose()  
#     print(t_a.value)

if __name__ == '__main__':
    N=5000
    d=load_mat("/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/2023_BC/bc001/raw/20230628/ms001mpt001_1.mat")
    history_volts=-(d["volts_raw_g"]-d["v_offset_g"])
    history_forces=np.zeros((N,3))
    history_forces_noisy = np.zeros((N, 3))
    history_forces_noisy_= np.zeros((N, 3))
    history_forces_butter = np.zeros((N, 3))
    before=-(d["volts_raw_g"]-d["v_offset_g"])
    history_time=d['t_g']
    history_pitch_is=d['pitch_is']
    history_Cp = np.zeros(N)
    history_dpitch_filtered=d['dpitch_filtered']
    history_phase=d['phase']
    fc_f = 20  # filter parameters
    order_f = 3
    fs = 550
    fc_p = 30
    order_p = 2
    b, a = signal.butter(order_f, fc_f / (fs / 2))  # to be tuned
    b_p, a_p = signal.butter(order_p, fc_p / (fs / 2))
    i = 0  # timestep counter

    ws_f=10
    ws=100
    i_list=[]
    for i in range(100,5000):
        window = history_volts[i - ws_f : i + 1]
        med = np.median(window, axis=0)

        # MAD = 1.4826 * np.median(
        MAD = 1.4826 * np.median(
            np.abs(window - med), axis=0
        )  # inspired from matlab filloutliers method

        mask = np.abs(history_volts[i] - med) > 3* MAD

        replace = np.where(
            mask, med + 3* MAD * np.sign(history_volts[i] - med), history_volts[i]
        )
        # replace = np.where(
        #     mask, med, history_volts[i]
        # )
        # print(pchip_interpolate(history_time[i-ws_f:i],history_volts[i-ws_f:i],history_time[i],axis=0))
        # s = InterpolatedUnivariateSpline(history_time[i-ws_f:i],history_volts[i-ws_f:i], k=2,axis=0)

        # replace = np.where(
        #     mask,pchip_interpolate(history_time[i-ws_f:i],history_volts[i-ws_f:i],history_time[i],axis=0), history_volts[i]
        # )
        # replace = np.where(
        #     mask, s(history_time[i]), history_volts[i]
        # )

        history_volts[i] = replace

        flatlined = False

        for l in range(5): #check if any of the volts channels flatlined
            if np.all(history_volts[i - ws_f : i + 1, l] == med[l]):
                flatlined = True
                history_volts[i - ws_f : i + 1, l] = before[i - ws_f : i + 1, l]
                # print(f'flatlined {l}',flush=True)
                i_list.append(i)
        
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

        
        flatlined = False

        for l in range(3): #check if any of the volts channels flatlined
            if np.all(history_forces_noisy_[i - ws_f : i + 1, l] == med[l]):
                flatlined = True
                history_forces_noisy_[i - ws_f : i + 1, l] = history_forces_noisy[i - ws_f : i + 1, l]
                print(f'flatlined {l}',flush=True)
                # i_list.append(i)



        # filtering step for forces
        history_forces_butter[i - ws : i + 1] = signal.filtfilt(
            b, a, history_forces_noisy_[i - ws : i + 1], axis=0,padlen=0
        )

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






    for k in range(5):            
        plt.figure()
        plt.plot(before[:i,k])
        plt.plot(history_volts[:i,k])

        plt.plot(i_list,np.zeros(len(i_list)),'o')
 
    for k in range(3):    
        plt.figure()
        plt.plot(history_forces_noisy[:i,k])
        plt.plot(history_forces_noisy_[:i,k])
        plt.plot(history_forces[:i,k])
    print(np.mean(history_Cp))
    plt.show() 