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

testing_dataset=torch.load("./NNet_files/test_set.pt")
test_index=np.loadtxt("./NNet_files/test_index.npy").astype(int)

mlp= MLPmodel.MLP(len(MLP_training.train_dataset.__getitem__(0)[0]),len(MLP_training.train_dataset.__getitem__(0)[1]),MLP_training.model,mean,std, m,tau)

mlp.load_state_dict(torch.load("./NNet_files/trained_net.pt")) #loads trained MLP
mlp.eval()


R2_memory=[]

for i,index in enumerate(test_index[:3]):
    nsteps=100
    
    
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
    # R2_test= sklearn.metrics.r2_score(Y_truth,Y_pred)
    print("\n Ytruth")
    print(np.array([testing_dataset.__getitem__(j)[0].detach().numpy() for j in range(index,index+nsteps+1)]))
    # R2_memory.append(R2_test)
    print("\n Y_pred")
    print(Y_pred)
    plt.figure()
    plt.plot(Y_truth[:,0])
    plt.plot(Y_pred[:,0])
    plt.plot(Y_pred2[:,0])
    
    
    plt.figure()
    plt.title('Cp')
    plt.plot(Y_truth[:,1])
    plt.plot(Y_pred[:,1])
    plt.plot(Y_pred2[:,1])
    
    
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
    
plt.show()

