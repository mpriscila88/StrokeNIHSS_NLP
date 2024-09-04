# ------------------------------------------------------
# Datasets with features obtained from LASSO
# ------------------------------------------------------

# This script:
    
# imports the coeficients for the variables obtained with LASSO regularization
# from the best fit in the ordinalNetTune in R, resulting in a reduced set
# of features for model training in R

# Both code and data are provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import re
import random
random.seed(42)

# path = path here
sys.path.insert(0, path) # insert path

a=pd.read_csv(os.path.join(path, 'coef13.csv'), sep='\s+')
a = a.iloc[1:,:].reset_index(drop=True)


x_train = pd.read_csv(os.path.join(path,'x_train_variables.csv'))
c=pd.DataFrame(x_train.columns, columns=['Features'])

ca = pd.concat([a,c], axis=1)

ca= ca[~(ca['logit(P[Y<=1])']==0)]

#------------------------------------------------------------------------
# Number of uni, bi and trig in the set of features
#------------------------------------------------------------------------

from nltk.tokenize import word_tokenize 

ca['n'] = ca.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

len(ca[ca.n == 1]) 
len(ca[ca.n == 2]) 
len(ca[ca.n == 3]) 


x_train=x_train[ca.Features]

x_test = pd.read_csv(os.path.join(path,'x_test_variables.csv'))

x_test=x_test[ca.Features]

x_train.to_csv(os.path.join(path,'xtrain_variables13.csv'),index=False)

x_test.to_csv(os.path.join(path,'xtest_variables13.csv'),index=False)


x_test = pd.read_csv(os.path.join(path,'x_validation_variables.csv'))

x_test=x_test[ca.Features]

x_test.to_csv(os.path.join(path,'x_validation_variables13.csv'),index=False)