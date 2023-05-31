import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import torch
import MLPmodel
from scipy import signal


# cols_to_keep = ['phase', 'pitch', 'Cp','Ct','Cr','Cm']
cols_to_keep_in = ['phase','pitch', 'Cp','Cr','Cm']
cols_to_keep_shift=['pitch', 'Cp','Cr','Cm']
# cols_to_keep_shift=[]
cols_to_keep_out = ['pitch', 'Cp','Cr','Cm']
m=2
tau=30

if __name__=="__main__":
    df=pd.read_pickle("../data/feedback_control_data_seb.pkl")
    
    #FILTERING STEP----------------------------------------
    
    b, a = signal.butter(8, 0.01)
    y = signal.filtfilt(b, a, df['Cm'].values, padlen=150)
    df['Cm']=y
    
    #----------------------------------------
    #NORMALIZATION STEP----------------------------------------
    # create numpy arrays to store the means and stds
    means = np.zeros((len(cols_to_keep_in),))
    stds = np.zeros((len(cols_to_keep_in),))

    # normalize each column
    for i, col in enumerate(cols_to_keep_in):
        
        # calculate the mean and standard deviation
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        # store the mean and standard deviation in the numpy arrays
        means[i] = col_mean
        stds[i] = col_std
        
        # normalize the column
        df[col] = (df[col] - col_mean) / col_std

    # save the means and stds to files
    np.savetxt('NNet_files/means.txt', means)
    np.savetxt('NNet_files/stds.txt', stds)
    #----------------------------------------

    all_Cpmean = df['Cp_mean']
    unique_Cp_mean = all_Cpmean.unique().tolist()
    
    # define the proportions of the three sets
    train_prop = 0.7
    val_prop = 0.15
    test_prop = 0.15

    # set the random seed for reproducibility

    random.seed(12)
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

    x_train=np.empty((0, len(cols_to_keep_in)+(m-1)*len(cols_to_keep_shift)+1))
    x_val=np.empty((0, len(cols_to_keep_in)+(m-1)*len(cols_to_keep_shift)+1))
    x_test=np.empty((0, len(cols_to_keep_in)+(m-1)*len(cols_to_keep_shift)+1))
    y_train=np.empty((0, len(cols_to_keep_out)))
    y_val=np.empty((0, len(cols_to_keep_out)))
    y_test=np.empty((0, len(cols_to_keep_out)))
    test_index=[0]
    list_max=[]
    list_min=[]
    list_mean=[]
    for i,value in enumerate(unique_Cp_mean[:]):
        print(i)
        # create a new dataframe that only includes rows with the current value
        sub_df = df[df['Cp_mean'] == value]


        sub_df=sub_df.iloc[11270:11270+3*1127,:]
        df_0=sub_df[cols_to_keep_shift][:-2*tau].reset_index(drop=True)
        df_tau=sub_df[cols_to_keep_in][tau:-tau].reset_index(drop=True)
        df_2tau=sub_df[cols_to_keep_out][2*tau:].reset_index(drop=True)
        df_diff_pitch=df_2tau["pitch"]-df_tau["pitch"]
        df_merged = pd.concat([df_tau,df_diff_pitch,df_0], axis=1)
        # list_max.append(np.max(df_diff_pitch))
        # list_min.append(np.min(df_diff_pitch))
        # list_mean.append(np.mean(df_diff_pitch))
        if i<1:
            np.savetxt("../data/phase.npy",np.array(df[df['Cp_mean'] == value][cols_to_keep_in].iloc[11270:11270+3*1127,:][tau:-tau].reset_index(drop=True))[::tau,0])
            #np.savetxt("../data/phase.npy",np.array(df_tau)[::tau,0])
        category=categories_list[i]
        
        if category =='train':
            x_train=np.vstack((x_train,df_merged.values))
            y_train=np.vstack((y_train,df_2tau.values))

        elif category =='test':
            x_test=np.vstack((x_test,df_merged.values))
            y_test=np.vstack((y_test,df_2tau.values))
            test_index.append(len(x_test))
            
        elif category =='val':
            x_val=np.vstack((x_val,df_merged.values))
            y_val=np.vstack((y_val,df_2tau.values))
    # print(list_max)
    # print(list_min)
    # print(list_mean)

    #USED TO GENERATE STARTING STATES FOR RL
    i_Cpmax=np.argmax(unique_Cp_mean)
    Cp_max=np.max(unique_Cp_mean)

    sub_df = df[df['Cp_mean'] == Cp_max]
    sub_df=sub_df.iloc[11270:11270+3*1127,:]
    df_0=sub_df[cols_to_keep_shift][:-2*tau].reset_index(drop=True)
    df_tau=sub_df[cols_to_keep_in][tau:-tau].reset_index(drop=True)
    df_2tau=sub_df[cols_to_keep_out][2*tau:].reset_index(drop=True)

    df_diff_pitch=df_2tau["pitch"]-df_tau["pitch"]
    df_merged = pd.concat([df_tau,df_diff_pitch,df_0], axis=1)


    print(f"training samples : {np.shape(x_train)[0]}")
    print(f"validation samples : {np.shape(x_val)[0]}")
    print(f"testing samples : {np.shape(x_test)[0]}")

    T_x_train=torch.tensor(x_train)
    T_x_val=torch.tensor(x_val)
    T_x_test=torch.tensor(x_test)
    T_y_train=torch.tensor(y_train)
    T_y_val=torch.tensor(y_val)
    T_y_test=torch.tensor(y_test)

    train_dataset=MLPmodel.Dataset(T_x_train,T_y_train)
    test_dataset=MLPmodel.Dataset(T_x_test,T_y_test)
    validation_dataset=MLPmodel.Dataset(T_x_val,T_y_val)
        
        
    torch.save(train_dataset,"./NNet_files/training_set.pt")
    torch.save(test_dataset,"./NNet_files/test_set.pt")
    torch.save(validation_dataset,"./NNet_files/validation_set.pt")
    np.savetxt("./NNet_files/test_index.npy",np.array(test_index))