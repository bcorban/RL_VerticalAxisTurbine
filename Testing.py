#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baptiste

This file uses the trained neural net on the test data and computes the associated R2s

"""

import torch
import matplotlib.pyplot as plt 
import MLPmodel 
import MLP_training
import numpy as np
import sklearn.metrics
from MLP_training import mean,std
from dataloader import m,tau

plot_bool=True
number_of_figs=5


testing_dataset=torch.load("./NNet_files/test_set.pt")
test_index=np.loadtxt("./NNet_files/test_index.npy").astype(int)

mlp= MLPmodel.MLP(len(MLP_training.train_dataset.__getitem__(0)[0]),len(MLP_training.train_dataset.__getitem__(0)[1]),MLP_training.model,mean,std, m,tau)

mlp.load_state_dict(torch.load("./NNet_files/trained_net.pt")) #loads trained MLP
mlp.eval()


R2_memory=[]

for i,index in enumerate(test_index[:-1]):
    nsteps=200
    
    
    # starting_state=testing_dataset.__getitem__(index)[0].float()
    # Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps+1)])
    # pitch_list=np.array([testing_dataset.__getitem__(j)[0].detach().numpy()[1] for j in range(index,index+nsteps+1)])
    # phase_list=np.array([testing_dataset.__getitem__(j)[0].detach().numpy()[0] for j in range(index,index+nsteps+1)])
    # Y_pred=mlp.predict_auto_regressive(starting_state,nsteps,pitch_list,phase_list)
    # Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps+1)])

    
    

    starting_state=np.array([testing_dataset.__getitem__(j)[0].float().detach().numpy() for j in range(index,index+(m-1)*tau)])
    Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps+1)])
    pitch_list=np.array([testing_dataset.__getitem__(j)[0].float().detach().numpy()[1] for j in range(index+(m-1)*tau,index+nsteps+1)])
    phase_list=np.array([testing_dataset.__getitem__(j)[0].float().detach().numpy()[0] for j in range(index+(m-1)*tau,index+nsteps+1)])
    Y_pred=mlp.predict_auto_regressive(starting_state,nsteps,pitch_list,phase_list)
    Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps+1)])

    R2_test= sklearn.metrics.r2_score(Y_truth[:-2],Y_pred)
    print(f"test n°{i} : R2 = {R2_test}")
    R2_memory.append(R2_test)
    # print("\n Ytruth")
    # print(np.array([testing_dataset.__getitem__(j)[0].detach().numpy() for j in range(index,index+nsteps+1)]))
    # print("\n Y_pred")
    # print(Y_pred)
    if plot_bool:
        if i<number_of_figs:
            # plt.figure()
            # plt.title("$\\alpha$")
            # plt.plot(Y_truth[:,0])
            # plt.plot(Y_pred[:,0])
            # plt.plot(Y_pred2[:,0])
            
            
            # plt.figure()
            # plt.title('Cp')
            # plt.plot(Y_truth[:,1])
            # plt.plot(Y_pred[:,1])
            # plt.plot(Y_pred2[:,1])
            
            
            # plt.figure()
            # plt.title('Cr')
            # plt.plot(Y_truth[:,2])
            # plt.plot(Y_pred[:,2])
            # plt.plot(Y_pred2[:,2])
            
            # plt.figure()
            # plt.title('Cm')
            # plt.plot(Y_truth[:,3])
            # plt.plot(Y_pred[:,3])
            # plt.plot(Y_pred2[:,3])
            
            fig,ax=plt.subplots(2,2)
            ax[0,0].set_title("$\\alpha$")
            ax[0,0].plot(Y_truth[:,0])
            ax[0,0].plot(Y_pred[:,0])
            ax[0,0].plot(Y_pred2[:,0])
            
            ax[0,1].set_title('Cp')
            ax[0,1].plot(Y_truth[:,1])
            ax[0,1].plot(Y_pred[:,1])
            ax[0,1].plot(Y_pred2[:,1])
            
            ax[1,0].set_title('Cr')
            ax[1,0].plot(Y_truth[:,2])
            ax[1,0].plot(Y_pred[:,2])
            ax[1,0].plot(Y_pred2[:,2])
            
            ax[1,1].set_title('Cm')
            ax[1,1].plot(Y_truth[:,3])
            ax[1,1].plot(Y_pred[:,3])
            ax[1,1].plot(Y_pred2[:,3])
            
            plt.tight_layout()
            
print(f"\n Average R2 on test dataset = {sum(R2_memory)/len(R2_memory)} \n")
plt.show()

