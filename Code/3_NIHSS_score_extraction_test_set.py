
# ------------------------------------------------------
# NIHSS score extraction from the test set notes
# ------------------------------------------------------

# This script imports notes ready for score extraction
# from 1_Notes_preprocessing_full_train_test_sets.py

# and adds the extracted scores to the test set obtained
# from 2_Train_test_split.py


# Only code is provided for reproducibility


import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta

# path = path here
sys.path.insert(0, path) # insert path

n = pd.read_csv(os.path.join(path,'notes_ready.csv')) # notes in the full train and test data (not provided - HPI)

# X_test only notes
X_test = pd.read_csv(os.path.join(path,'x_test_outcomes.csv'))

n = pd.merge(n.drop(columns='nihss_score'), X_test[['PatientID', 'jc_admitdate', 'gs_discdatetime', 'nihss_score']], on=['PatientID', 'jc_admitdate', 'gs_discdatetime'])

n = n.sort_values(['ContactDTS',"NoteID"], ascending=[True,True])

n = n[['PatientID', 'SexDSC', 'Age', 'jc_admitdate', 'gs_discdatetime', 'nihss_score','NoteID','Notes']].drop_duplicates().reset_index(drop=True)

# add spaces
n.Notes = ' ' + n.Notes.astype(str) + ' '

n = n.groupby(['PatientID', 'SexDSC', 'Age', 'jc_admitdate', 'gs_discdatetime', 'nihss_score']).Notes.sum().reset_index()


# After merging the notes per admission, extract scores


from extract_scores import extract_scores

s = extract_scores(n)                   

s = s[['PatientID','jc_admitdate', 'nihss_score','mode_score', 'min_score', 'max_score']]

X_test = pd.read_csv(os.path.join(path,'x_test_outcomes.csv'))

s.jc_admitdate = s.jc_admitdate.astype('datetime64[ns]')
X_test.jc_admitdate = X_test.jc_admitdate.astype('datetime64[ns]')
X_test = pd.merge(X_test,s,on=['PatientID','jc_admitdate'])

X_test.to_csv(os.path.join(path,'x_test_outcomes.csv'))

