
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_encode(df, outcome, feature):
    
    #------------------------------------------------------------------------
    # Train/test dataset
    #------------------------------------------------------------------------
    
    #Split patients in train/test #################################################

    e = df[['PatientID']].drop_duplicates()

    e_train, e_test = train_test_split(e, test_size=0.3, random_state=42) 

    # Assign all respective MRNs encounters in train and test
    df_train =  df[df.PatientID.isin(e_train.PatientID)] 
    df_test =  df[df.PatientID.isin(e_test.PatientID)] 


    return df_train, df_test

