
# ------------------------------------------------------
# Notes Preprocessing for the validation set
# ------------------------------------------------------

# This script imports the model train set from 2_Train_test_split.py
# to use as training vocabulary for vectorization of the validation 
# set notes into a structured format 

# Only code is provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import re
import random
random.seed(42)

# path = path here
sys.path.insert(0, path) # insert path

from traintestencode import train_test_encode
from filterfeatures import filter_features
from ngram_vec import ngram
from dateutil.relativedelta import relativedelta

#------------------------------------------------------------------------
# Load train data
#------------------------------------------------------------------------
 
X_train = pd.read_csv(os.path.join(path,'x_train_outcomes.csv')) # in the original version this matrix comes with a column "Notes" (not provided - HPI)

x_train = pd.read_csv(os.path.join(path,'x_train_variables.csv'))

#------------------------------------------------------
# Load validation data
#------------------------------------------------------

n = pd.read_csv(os.path.join(path,'validation_set_notes.csv')) # notes in the validation set (not provided - HPI)

n.Notes = n.TEXT

#------------------------------------------------------
# Notes preprocessing
#------------------------------------------------------

def process_test_data(X_train, n, feature):

    from text_preprocessing import process_text
        
    n = process_text(n)
    
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
    
    from ngram_vec import ngram
    
    feature = 'Notes'
    X_test = n
    
    # vectorization for combinations of n-grams
    
    x_train11, x_test11 = ngram(X_train, X_test, feature, ngram_range=(1, 1))
    print(1)
    x_train12, x_test12 = ngram(X_train, X_test, feature, ngram_range=(1, 2))
    print(2)
    x_train13, x_test13 = ngram(X_train, X_test, feature, ngram_range=(1, 3))
    print(3)
    x_train22, x_test22 = ngram(X_train, X_test, feature, ngram_range=(2, 2))
    print(4)
    x_train23, x_test23 = ngram(X_train, X_test, feature, ngram_range=(2, 3))
    print(5)
    x_train33, x_test33 = ngram(X_train, X_test, feature, ngram_range=(3, 3))
    print(6)
    
    #join all    
    x_tr = pd.concat([x_train11, x_train12, x_train13, x_train22, x_train23, x_train33], axis = 1)
    x_t = pd.concat([x_test11, x_test12, x_test13, x_test22, x_test23, x_test33], axis = 1)   
    
    #remove repeated columns
    columns = x_tr.columns.drop_duplicates()
    x_train = x_tr.loc[:,~x_tr.columns.duplicated()]
    x_test = x_t.loc[:,~x_t.columns.duplicated()]
    
    x_train[x_train >0] = 1 
    x_test[x_test >0] = 1 
    
    return x_test

x_test = process_test_data(X_train, n, feature)

x_test = x_test[x_train.columns]

x_test.to_csv(os.path.join(path,'x_validation_variables.csv'), index=False)



