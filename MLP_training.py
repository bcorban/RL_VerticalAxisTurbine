"""
This file trains the feed-forward network and is where the optimizer parameters and the architecture are entered.
"""

import torch
import torch.nn as nn
import MLPmodel 
import numpy as np
from dataloader import m,tau
#Training parameters
lr=1e-4
epoch=50
model=[128,128,128]
batch_size=512

#Dataset preparation

train_dataset=torch.load("./NNet_files/training_set.pt")
validation_dataset=torch.load("./NNet_files/validation_set.pt")
mean=np.loadtxt("NNet_files/means.txt") 
std=np.loadtxt("NNet_files/stds.txt")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)


# Initialize the MLP
mlp = MLPmodel.MLP(len(train_dataset.__getitem__(0)[0]),len(train_dataset.__getitem__(0)[1]),model,mean,std, m,tau)


# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr)


#Training
if __name__=="__main__":
    mlp.Train(model,epoch, train_loader, validation_loader, optimizer, loss_function)

    torch.save(mlp.state_dict(),"./NNet_files/trained_net.pt")

