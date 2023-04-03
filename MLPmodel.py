#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baptiste

This file creates the classes for the feed forward, and the different datasets necessary
"""
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
import time
import numpy as np

class MLP(nn.Module):
    def __init__(self,n_input,n_out,model,mean,std,m,tau):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(n_input, model[0]))
        self.layers.append(nn.ReLU())
        for layer in range(len(model)-1):
            self.layers.append(nn.Linear(model[layer],model[layer+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(model[-1],n_out))
        # self.print_model()
        self.mean=mean
        self.std=std
        self.m=m
        self.tau=tau

    def forward(self, x):
     
        out=self.layers(x.float())
        return out
    
    
    def print_model(self):
        print(self.layers)


    def Train(self,model, n_epoch, train_loader, validation_loader, optimizer, loss_function, plot=True):
        # Run the training loop
        start_time=time.time()
        loss_history_tr=[]
        loss_history_validation=[]
        patience=0
        
        for epoch in range(0, n_epoch): 
            
            # Iterate over the DataLoader for training data
            for i, datas in enumerate(train_loader, 0):
            #   print(i)
              # Get and prepare inputs
              inputs, targets = datas
              inputs, targets = inputs.float(), targets.float()
              
              # Zero the gradients
              optimizer.zero_grad()
              
              # Perform forward pass
              outputs =self.forward(inputs)
             
              # Compute loss
              loss = loss_function(outputs, targets)
              
              
              # Perform backward pass
              loss.backward()
              
              # Perform optimization
              optimizer.step()
              
              
            loss_history_tr.append(float(loss)) #add loss at the given epoch to loss history
            
            
            loss_validation=0
            for j, validation in enumerate(validation_loader, 0): #compute mean loss over the validation dataset
                inputs_t, targets_t = validation
                inputs_t, targets_t = inputs_t.float(), targets_t.float()
                outputs_t=self.forward(inputs_t)
                loss_validation+=loss_function(outputs_t,targets_t)
            loss_validation=loss_validation/j
            loss_history_validation.append(float(loss_validation))
            
            print("Epoch : %d - Train_loss = %f - validation_loss=%f" %(epoch+1,loss,loss_validation))
            
            
            #Early stopping if no improvement, to prevent overfitting
            if epoch>10:
                if loss_history_validation[-1]>loss_history_validation[-2]:
                    patience+=1
                if loss_history_validation[-1]<loss_history_validation[-2]:
                    patience=0
                
            if patience==5:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            
        print(f"Training duration : {time.time()-start_time}")
            
            
            
        #plot training history
        if plot:
            epoch_list=[i for i in range(0,epoch+1)]
            plt.figure()
            ax = plt.subplot(111)
            plt.semilogy(epoch_list,loss_history_tr,label="Training set")
            plt.semilogy(epoch_list,loss_history_validation, label="Validation")
            plt.grid()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training {model}")
            # plt.ylim(1e-4,1)
            plt.show()

    
    def predict_auto_regressive(self,starting_state,n_steps,pitch_list,phase_list):
        if self.m==1:
            
            state_in=starting_state.numpy()
            history=np.empty((0, len(state_in)-2))
            for s in range(n_steps-1):
                print("in")
                print(state_in)
                print("out")
                state_out=self.__call__(torch.tensor(state_in).float()).detach().numpy()
                print(state_out)
                history=np.vstack((history,state_out))
                dalpha=pitch_list[s+1]-state_out[0]
                state_in[1:len(state_in)-1]=state_out
                state_in[-1]=dalpha
                state_in[0]=phase_list[s+1]
            print(history)
            return history
        elif self.m==2:
            print('Starting_state:')
            print(starting_state)
            state_in=starting_state[-1]
            history=np.empty((0, 4))
            for state in starting_state[1:]:
                history=np.vstack((history,state[1:5]))
            print('history')
            print(history)            
            for s in range(n_steps-1):
                print("in")
                print(state_in)
                print("out")
                state_out=self.__call__(torch.tensor(state_in)).detach().numpy()
                print(state_out)
                history=np.vstack((history,state_out))
                dalpha=pitch_list[s+1]-state_out[0]
                state_in=np.zeros(len(state_in)) #to be deleted, temporary to check if any 0 remain
                state_in[1:len(state_out)+1]=state_out
                state_in[len(state_out)+1]=dalpha
                state_in[0]=phase_list[s+1]
                state_in[len(state_out)+2:]=history[-self.tau][:2]
                print('history -tau')
                print(history[-self.tau])
                state_in.astype(float)
            
            return history
        

#-----------------------------------------------------------------------------


class Dataset(torch.utils.data.Dataset): #class dataset used in the snapshot creation
    # load the dataset
    def __init__(self,X,y):
        # store the inputs and outputs
        self.X = X
        self.y = y
 
    # number of rows in the dataset
    def __len__(self):
        return self.X.size()[0]
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 