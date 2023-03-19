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
import matplotlib as mpl


testing_dataset=torch.load("./NNet_files/test_set.pt")

mlp= MLPmodel.MLP(len(MLP_training.train_dataset.__getitem__(0)[0]),len(MLP_training.train_dataset.__getitem__(0)[1]),MLP_training.model)

mlp.load_state_dict(torch.load("/NNet_files/trained_net.pt")) #loads trained mlp
mlp.eval()

mean=np.loadtxt("NNet_files/means.txt") #useful to de-standardize the output
std=np.loadtxt("NNet_files/stds.txt")


R2_memory=[]

# for j in range(len(validation_index)-1):
    
#     Ypred=[mlp(validation_dataset.__getitem__(i)[0].float()).detach().numpy() for i in range(int(validation_index[j][0]),int(validation_index[j+1][0]))]
#     Ytruth=[validation_dataset.__getitem__(i)[1].detach().numpy() for i in range(int(validation_index[j][0]),int(validation_index[j+1][0]))]
#     Ypred=np.array(Ypred)*rms+mean
#     Ytruth=np.array(Ytruth)*rms+mean
#     error=(Ytruth-Ypred)/Ytruth
#     t=[i for i in range(int(validation_index[j+1][0])-int(validation_index[j][0]))]
#     R2_test= sklearn.metrics.r2_score(Ytruth,Ypred)
    
#     R2_memory.append(R2_test)
    
    
#     if display_plots: #subplot with the chosen sims
#         if j in displayed_sim:
            
#             axs[displayed_sim.index(j)].plot(t,Ytruth, label="Simulation",c="#3491AF")
#             axs[displayed_sim.index(j)].plot(t,Ypred,label="Neural network prediction",linestyle="dashed",c="#E94F06")
#             # axs[displayed_sim.index(j)].plot(t,error, label="Error",alpha=0.8)
#             if j==0:
#                 axs[displayed_sim.index(j)].set_title("Hawkmoth",fontsize=18)
#             elif j==1:
#                 axs[displayed_sim.index(j)].set_title("Honeybee",fontsize=18)
#             elif j==2:
#                 axs[displayed_sim.index(j)].set_title("Fruit Fly",fontsize=18)
#                 print((sum(Ytruth)-sum(Ypred))/sum(Ytruth))
#             else:
#                 axs[displayed_sim.index(j)].set_title(f"a1={validation_index[j][1]:.3f}, b1={validation_index[j][2]:.3f}, b3={validation_index[j][3]:.3f}, $\\varphi$={validation_index[j][4]:.2f}, $\\delta$={validation_index[j][5]}")
#             # axs[displayed_sim.index(j)].set_ylabel(output_name,fontsize=14)
#             # axs[displayed_sim.index(j)].set_xlim([0,492])
#         # if j==0:
#         #     axs[j].legend(fontsize=14)
#         # axs[0].legend(loc='upper right',fontsize=18)
        
        
#         if display_all:
#             f,ax = plt.subplots(1,2,figsize=(10,4))
#             f.tight_layout()
#             ax[0].plot(t,Ytruth, label="Truth")
#             ax[0].plot(t,Ypred,label="Prediction")
#             ax[0].plot(t,error, label="Error")
#             ax[0].grid()
            
#             box = ax[0].get_position()
#             ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
#             # Put a legend to the right of the current axis
#             ax[0].legend(loc='lower left',fontsize=7)
            
#             ax[0].set_xlabel("Timesteps")
#             ax[0].set_ylabel(output_name)
        
            
#             ax[1].plot(Ytruth, Ypred, 'k.')
#             ax[1].plot(Ytruth, Ytruth, 'r--')
#             ax[1].set_xlabel("Truth")
#             ax[1].set_ylabel("Prediction")
#             plt.setp(ax[1], xticks=[], yticks=[])
#             plt.suptitle(f"Validation set n°{j+1}, a1={validation_index[j][1]:.3f}, b1={validation_index[j][2]:.3f}, b3={validation_index[j][3]:.3f}, $\\varphi$={validation_index[j][4]:.2f}, $\\delta$={validation_index[j][5]}")
#             f.tight_layout()
    
#     print(f'The coefficient of determination for test set n°{j} is R2 = {R2_test}')
    
# print(f"\n \n meanR2={sum(R2_memory)/len(R2_memory)}")



# fig.savefig(f"./Validation_plots/{output}_validation_R2_{sum(R2_memory)/len(R2_memory)}.eps",bbox_inches='tight')
# np.save("./NNet_files_f/R2.npy",np.array(R2_memory))
# if display_plots:
#     R2_trunq=[max(R2,0.85) for R2 in R2_memory]
#     ax.scatter(CL,np.array(CL)/-np.array(CP),c=R2_trunq,cmap=cmp)
#     ax.set_ylabel("$\\eta$")
#     ax.set_xlabel("$C_L$")
#     norm = mpl.colors.Normalize(vmin=np.array(R2_trunq).min(), vmax=np.array(R2_trunq).max())

#     cb=figu.colorbar(cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax,ticks=np.linspace(np.array(R2_trunq).min(), np.array(R2_trunq).max(), 8, endpoint=True))
#     cb.ax.set_xlabel('R2')
#     ax.grid(alpha=0.2)

#     figu.tight_layout()



