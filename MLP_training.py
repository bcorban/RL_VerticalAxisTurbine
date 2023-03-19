"""
This file trains the feed-forward network and is where the optimizer parameters and the architecture are entered.
"""

import torch
import torch.nn as nn
import MLPmodel 

#Training parameters
lr=1e-3
epoch=50
model=[16,32,16]
batch_size=256

#Dataset preparation

train_dataset=torch.load("./NNet_files/training_set.pt")
validation_dataset=torch.load("./NNet_files/validation_set.pt")


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)


# Initialize the MLP
mlp = MLPmodel.MLP(len(train_dataset.__getitem__(0)[0]),len(train_dataset.__getitem__(0)[1]),model)


# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr)


#Training
mlp.Train(model,epoch, train_loader, validation_loader, optimizer, loss_function)

torch.save(mlp.state_dict(),"./NNet_files/trained_net.pt")

