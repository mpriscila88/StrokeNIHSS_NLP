# ------------------------------------------------------
# Train test splitting
# ------------------------------------------------------

# This script imports the preprocessed notes for each admission 
# from 1_Notes_preprocessing_full_train_test_sets.py and creates train and test sets

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
# Load data
#------------------------------------------------------------------------
 
n = pd.read_csv(os.path.join(path,'df_with_lemma.csv')) # notes preprocessed (not provided - HPI)

feature = 'Notes'
outcome = 'nihss_score' 

# Count tokens   
n['Ntokens'] = n.Notes.astype('str').apply(lambda x: len(re.findall(r'\w+',x)))

n = n[n['Ntokens'] > 250] 


# Plot score distribution

n2 = n[['PatientID', 'jc_admitdate',
       'gs_discdatetime', outcome]].drop_duplicates() 

n2['NIHSS'] = n2[outcome]

import matplotlib.pyplot as plt
import seaborn as sns
 

fig, ax = plt.subplots(figsize=(20,5))
labels, counts = np.unique(n2.NIHSS, return_counts=True)
plt.bar(labels, counts, align='center')
plt.xticks(np.arange(0, 42+1, 1))
plt.xlabel('NIHSS')
plt.ylabel('Frequency')
ax.set_xlim(-0.5,42)

#------------------------------------------------------
# Create train and test sets
#------------------------------------------------------

#stratified random sampling 
X_train, X_test = train_test_encode(n, outcome, feature)

#------------------------------------------------------------------------
# Remove misrepresented features in train
#------------------------------------------------------------------------
 
X_train2 = filter_features(X_train, outcome, feature, path)

X_train = pd.concat([X_train.drop(columns='Notes'),X_train2[['Notes']]], axis=1)

#------------------------------------------------------------------------
# Merge notes
#------------------------------------------------------------------------

def merge(d):
    d = d.sort_values(['jc_admitdate',"NoteID"], ascending=[True,True])

    #add spaces
    d.Notes = ' ' + d.Notes.astype(str) + ' '

    d = d.groupby(['PatientID','SexDSC','Age','jc_admitdate','gs_discdatetime','nihss_score'
                    ]).Notes.sum().reset_index()
 
    d.Notes = d.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces

    return d

X_train = merge(X_train)
X_test = merge(X_test)

y_train = X_train.nihss_score
y_test = X_test.nihss_score


# save X_train and X_test

X_train.to_csv(os.path.join(path,'x_train_outcomes.csv'), index=False)
X_test.to_csv(os.path.join(path,'x_test_outcomes.csv'), index=False)


#------------------------------------------------------------------------
# Vectorization
#------------------------------------------------------------------------

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


# save x_train and x_test

x_train.to_csv(os.path.join(path,'x_train_variables.csv'), index=False)
x_test.to_csv(os.path.join(path,'x_test_variables.csv'), index=False)
