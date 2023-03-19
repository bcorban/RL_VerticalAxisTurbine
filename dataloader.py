import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import torch
df=pd.read_pickle("feedback_control_data.pkl")

all_Cpmean = df['Cp_mean']
unique_Cp_mean = all_Cpmean.unique().tolist()

 
# define the proportions of the three sets
train_prop = 0.7
val_prop = 0.15
test_prop = 0.15

# set the random seed for reproducibility
random.seed(123)

# calculate the number of elements in each category based on the proportions
train_count = int(train_prop * len(unique_Cp_mean))
val_count = int(val_prop * len(unique_Cp_mean))
test_count = int(test_prop * len(unique_Cp_mean))

# create a list to store the category assignments
categories_list = []

# assign each value in the original list to a category randomly
for i in range(len(unique_Cp_mean)):
    if train_count > 0:
        categories_list.append('train')
        train_count -= 1
    elif val_count > 0:
        categories_list.append('val')
        val_count -= 1
    else:
        categories_list.append('test')
        test_count -= 1

# shuffle the categories list randomly
random.shuffle(categories_list)

x_train=np.empty((0, 7))
x_val=np.empty((0, 7))
x_test=np.empty((0, 7))
y_train=np.empty((0, 6))
y_val=np.empty((0, 6))
y_test=np.empty((0, 6))

for i,value in enumerate(unique_Cp_mean[:1]):
    # create a new dataframe that only includes rows with the current value
    sub_df = df[df['Cp_mean'] == value]
    cols_to_keep = ['phase', 'pitch', 'Cp','Ct','Cr','Cm']

    # create a new dataframe with only the selected columns and without the first row
    df_no_first = df[cols_to_keep][1:]
    df_no_first = df_no_first.reset_index(drop=True)
    # create a new dataframe with only the selected columns and without the last row
    df_no_last = df[cols_to_keep][:-1]
    
    #add column containing the 'pitch command'
    
    df_no_last['pitch_increment'] = df['pitch'].diff(periods=-1)[:-1]
    df_no_last['pitch_increment'] = -1*df_no_last['pitch_increment'] 
    
    category=categories_list[i]

    if category =='train':
        x_train=np.vstack((x_train,df_no_last.values))
        y_train=np.vstack((y_train,df_no_first.values))

    if category =='test':
        x_test=np.vstack((x_test,df_no_last.values))
        y_test=np.vstack((y_test,df_no_first.values))
        
    if category =='val':
        x_val=np.vstack((x_val,df_no_last.values))
        y_val=np.vstack((y_val,df_no_first.values))

T_x_train=torch.tensor(x_train)
T_x_val=torch.tensor(x_val)
T_x_test=torch.tensor(x_test)
T_y_train=torch.tensor(y_train)
T_y_val=torch.tensor(y_val)
T_test=torch.tensor(y_test)