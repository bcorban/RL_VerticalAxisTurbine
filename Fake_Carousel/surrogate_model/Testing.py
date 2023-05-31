#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baptiste

This file uses the trained neural net on the test data and computes the associated R2s

"""


import numpy as np
import torch
import matplotlib.pyplot as plt 
import MLPmodel 
import MLP_training

import sklearn.metrics
from MLP_training import mean,std
from dataloader import m,tau
# import os
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
plot_bool=True
number_of_figs=0


testing_dataset=torch.load("./NNet_files/test_set.pt")
test_index=np.loadtxt("./NNet_files/test_index.npy").astype(int)

mlp= MLPmodel.MLP(len(MLP_training.train_dataset.__getitem__(0)[0]),len(MLP_training.train_dataset.__getitem__(0)[1]),MLP_training.model,mean,std, m,tau)

mlp.load_state_dict(torch.load("./NNet_files/trained_net.pt")) #loads trained MLP
mlp.eval()


R2_memory=[]
Cp_mean_memory=[]
phase_list=np.loadtxt("../data/phase.npy")
T=1127
nsteps=1*int(T/tau)

print(f"number of steps performed : {nsteps}")


for i,index in enumerate(test_index[:-1]):

    assert(index+nsteps*tau<test_index[i+1])
    starting_state=testing_dataset.__getitem__(index)[0].float().detach().numpy()
    Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps*tau)])

    pitch_list=np.array([testing_dataset.__getitem__(index+j*tau)[0].float().detach().numpy()[1] for j in range(0,nsteps)])
    # print(pitch_list)
    Y_pred=mlp.predict_auto_regressive(starting_state,nsteps,pitch_list,phase_list)

    Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps*tau+1)])

    R2_test= sklearn.metrics.r2_score(Y_truth[::tau,1][:-2],Y_pred[1:,1])
    print(f"test n°{i} : R2 = {R2_test}")
    R2_memory.append(R2_test)
    Cp_mean_memory.append(Y_truth[:,1].mean())

    if plot_bool:
        if i==230:

            fig,ax=plt.subplots(2,2)
            fig.suptitle(f"R2={R2_test}")
            ax[0,0].set_title("$\\alpha$")
            ax[0,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,0])
            ax[0,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,0])
            ax[0,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,0])
            
            ax[0,1].set_title('Cp')
            ax[0,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,1])
            ax[0,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,1])
            ax[0,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,1])
            
            ax[1,0].set_title('Cr')
            ax[1,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,2])
            ax[1,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,2])
            ax[1,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,2])
            
            ax[1,1].set_title('Cm')
            ax[1,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,3])
            ax[1,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,3])
            ax[1,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,3])
            
            plt.tight_layout()
            plt.show()
plt.figure()
plt.scatter(Cp_mean_memory,R2_memory,marker='+')
# plt.show()
best=np.argsort(R2_memory)[-5:]
worst=np.argsort(R2_memory)[:5]

idx_1=np.argwhere(np.array(Cp_mean_memory)>0.15)
ix=np.argmax(np.array(R2_memory)[idx_1])
best_ind=idx_1[ix]
print(best_ind)
plt.scatter(Cp_mean_memory[best_ind[0]],R2_memory[best_ind[0]],c='red')
print(testing_dataset.__getitem__(test_index[best_ind[0]])[0].detach().numpy())

for i in best:
    index=test_index[i]

    starting_state=testing_dataset.__getitem__(index)[0].float().detach().numpy()
    Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps*tau+1)])
    pitch_list=np.array([testing_dataset.__getitem__(index+j*tau)[0].float().detach().numpy()[1] for j in range(0,nsteps)])

    Y_pred=mlp.predict_auto_regressive(starting_state,nsteps,pitch_list,phase_list)

    Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps*tau+1)])

    R2_test= R2_memory[i]
    print(f"test n°{i} : R2 = {R2_test}")
 

    fig,ax=plt.subplots(2,2)
    fig.suptitle(f"R2={R2_test}")
    ax[0,0].set_title("$\\alpha$")
    ax[0,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,0])
    ax[0,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,0])
    ax[0,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,0])
    
    ax[0,1].set_title('Cp')
    ax[0,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,1])
    ax[0,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,1])
    ax[0,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,1])
    
    ax[1,0].set_title('Cr')
    ax[1,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,2])
    ax[1,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,2])
    ax[1,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,2])
    
    ax[1,1].set_title('Cm')
    ax[1,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,3])
    ax[1,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,3])
    ax[1,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,3])
    
    plt.tight_layout()

for i in worst:
    index=test_index[i]

    starting_state=testing_dataset.__getitem__(index)[0].float().detach().numpy()
    Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps*tau+1)])
    pitch_list=np.array([testing_dataset.__getitem__(index+j*tau)[0].float().detach().numpy()[1] for j in range(0,nsteps)])
    Y_pred=mlp.predict_auto_regressive(starting_state,nsteps,pitch_list,phase_list)

    Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps*tau+1)])

    R2_test= R2_memory[i]
    print(f"test n°{i} : R2 = {R2_test}")


    fig,ax=plt.subplots(2,2)
    fig.suptitle(f"R2={R2_test}")
    ax[0,0].set_title("$\\alpha$")
    ax[0,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,0])
    ax[0,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,0])
    ax[0,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,0])
    
    ax[0,1].set_title('Cp')
    ax[0,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,1])
    ax[0,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,1])
    ax[0,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,1])
    
    ax[1,0].set_title('Cr')
    ax[1,0].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,2])
    ax[1,0].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,2])
    ax[1,0].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,2])
    
    ax[1,1].set_title('Cm')
    ax[1,1].plot([n/T for n in range(tau,tau+len(Y_truth))],Y_truth[:,3])
    ax[1,1].plot([n*tau/T for n in range(nsteps-1)],Y_pred[:,3])
    ax[1,1].plot([n/T for n in range(tau,tau+len(Y_pred2))],Y_pred2[:,3])
    plt.tight_layout()




print(f"\n Average R2 on test dataset = {sum(R2_memory)/len(R2_memory)} \n")
plt.show()

