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
tau=10

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
    df=df[::5]
    all_Cpmean = df['Cp_mean']
    unique_Cp_mean = all_Cpmean.unique().tolist()
    
    # define the proportions of the three sets
    train_prop = 0.7
    val_prop = 0.15
    test_prop = 0.15

    # set the random seed for reproducibility
    # random.seed(123)
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

    for i,value in enumerate(unique_Cp_mean[:]):
        # print(i)
        # create a new dataframe that only includes rows with the current value
        sub_df = df[df['Cp_mean'] == value]
        
        sub_df=sub_df.iloc[2000:2400,:]



        # create a new dataframe with only the selected columns and without the last row
        df_no_last = sub_df[cols_to_keep_in][:-1]

        #add column containing the 'pitch command'
        
        df_no_last['pitch_increment'] = sub_df['pitch'].diff(periods=-1)[:-1]
        df_no_last['pitch_increment'] = -1*df_no_last['pitch_increment'] 
        if m==2:
            df_no_last_shift=df_no_last.iloc[tau:].copy()
            df_no_last_0 = df_no_last.iloc[:-tau].copy()
            df_no_last_0= df_no_last_0.drop('pitch_increment', axis=1)
            df_no_last_0=df_no_last_0[cols_to_keep_shift]
            df_merged = pd.concat([df_no_last_shift.reset_index(drop=True),df_no_last_0.reset_index(drop=True)], axis=1)
            df_no_first = sub_df[cols_to_keep_out][tau+1:]
            df_no_first = df_no_first.reset_index(drop=True)
            category=categories_list[i]
        elif m==1:
            df_merged=df_no_last
            df_no_first = sub_df[cols_to_keep_out][1:]
            df_no_first = df_no_first.reset_index(drop=True)
            category=categories_list[i]

        if category =='train':
            x_train=np.vstack((x_train,df_merged.values))
            y_train=np.vstack((y_train,df_no_first.values))

        elif category =='test':
            x_test=np.vstack((x_test,df_merged.values))
            y_test=np.vstack((y_test,df_no_first.values))
            test_index.append(len(x_test))
            
        elif category =='val':
            x_val=np.vstack((x_val,df_merged.values))
            y_val=np.vstack((y_val,df_no_first.values))

    #USED TO GENERATE STARTING STATES FOR RL
    
    # Cp_max=np.max(unique_Cp_mean)
    # sub_df = df[df['Cp_mean'] == Cp_max]
        
    # sub_df=sub_df.iloc[2000:2400,:]
    # print(sub_df.values[:10])
    # # create a new dataframe with only the selected columns and without the last row
    # df_no_last = sub_df[cols_to_keep_in][:-1]

    # #add column containing the 'pitch command'
    
    # df_no_last['pitch_increment'] = sub_df['pitch'].diff(periods=-1)[:-1]
    # df_no_last['pitch_increment'] = -1*df_no_last['pitch_increment'] 
    # if m==2:
    #     df_no_last_shift=df_no_last.iloc[tau:].copy()
    #     df_no_last_0 = df_no_last.iloc[:-tau].copy()
    #     df_no_last_0= df_no_last_0.drop('pitch_increment', axis=1)
    #     df_no_last_0=df_no_last_0[cols_to_keep_shift]
    #     df_merged = pd.concat([df_no_last_shift.reset_index(drop=True),df_no_last_0.reset_index(drop=True)], axis=1)
    # print(df_merged.values[:10])
    # np.savetxt("./data/starting_history.npy",np.array(df_merged.values)[:10])


    
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