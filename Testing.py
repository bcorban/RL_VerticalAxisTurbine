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

for i,index in enumerate(test_index[:2]):
    nsteps=test_index[i+1]-index
    starting_state=testing_dataset.__getitem__(index)[0].float()
    Y_truth=np.array([testing_dataset.__getitem__(j)[1].detach().numpy() for j in range(index,index+nsteps)])
    Y_pred=mlp.predict_auto_regressive(starting_state,nsteps)
    Y_pred2=np.array([mlp(testing_dataset.__getitem__(j)[0].float()).detach().numpy() for j in range(index,index+nsteps)])
    # R2_test= sklearn.metrics.r2_score(Y_truth,Y_pred)
    print("\n Ypred2")
    print(Y_pred2[:3,:])
    # R2_memory.append(R2_test)

    plt.figure()
    plt.plot(Y_truth[:,0])
    plt.plot(Y_pred[:,0])
    plt.plot(Y_pred2[:,0])
    
    plt.figure()
    plt.plot(Y_truth[:,1])
    plt.plot(Y_pred[:,1])
    plt.plot(Y_pred2[:,1])
    
    plt.figure()
    plt.plot(Y_truth[:,2])
    plt.plot(Y_pred[:,2])
    plt.plot(Y_pred2[:,2])
    
plt.show()
print(R2_memory)

#add std to the MLPmodel