

# -------------------------------------------------------
# NIHSS score extraction from the validation set notes
# -------------------------------------------------------

# This script imports the validation set notes for score extraction

# and adds the extracted scores to the expert-revised validation set

# Only code is provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta

# path = path here
sys.path.insert(0, path) # insert path

# import os, gzip, shutil

dir_name =  path+'mimic-iii-clinical-database-1.4/'

# def gz_extract(directory):
#     extension = ".gz"
#     os.chdir(directory)
#     for item in os.listdir(directory): # loop through items in dir
#       if item.endswith(extension): # check for ".gz" extension
#           gz_name = os.path.abspath(item) # get full path of files
#           file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] #get file name for file within
#           with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
#               shutil.copyfileobj(f_in, f_out)
#           # os.remove(gz_name) # delete zipped file
        
# gz_extract(dir_name)

#------------------------------------------------------
# Load data
#------------------------------------------------------

df = pd.read_csv(os.path.join(dir_name,'NOTEEVENTS.csv')) 

path0 = path+'national-institutes-of-health-stroke-scale-nihss-annotations-for-the-mimic-iii-database-1.0.0/'
sys.path.insert(0, path0) # insert path


train=pd.read_table(os.path.join(path0,"NER_Train.txt"),names=['X'])

test=pd.read_table(os.path.join(path0,"NER_Test.txt"),names=['X']) 

text = pd.concat([train,test], axis=0).reset_index(drop=True)
 
train['HADM_ID'] = train.X.apply(lambda x: re.findall('HADM_ID\': (.+?),', x))

train = pd.DataFrame(list(train['HADM_ID'])).T
train.columns=['HADM_ID']

test['HADM_ID'] = test.X.apply(lambda x: re.findall('HADM_ID\': (.+?),', x))

test = pd.DataFrame(list(test['HADM_ID'])).T
test.columns=['HADM_ID']
    
ids = pd.concat([train,test], axis=0).reset_index(drop=True)

df.HADM_ID = df.HADM_ID.astype(float)
 
df = df[df.HADM_ID.isin(ids.HADM_ID.astype(float))]

df = df[df.CATEGORY == 'Discharge summary']

# Sort by Line number
df = df.sort_values(["ROW_ID"], ascending=[True])

df.TEXT = ' ' + df.TEXT.astype(str) + ' '

df = df.groupby(['SUBJECT_ID','HADM_ID']).TEXT.sum().reset_index()

df['Notes'] = df['TEXT']

# After merging the notes per admission, extract scores

# -----------------------------------------------------------------------------
# Extract NIHSS based on rules (Best performance for NIHSS extraction in paper)
# -----------------------------------------------------------------------------

from extract_scores import extract_scores

s = extract_scores(df)

x_test = pd.read_csv(os.path.join(path,'x_validation_revised.csv')) # not provided - HPI

n = pd.merge(x_test[['SUBJECT_ID', 'HADM_ID', 'nihss_score']], 
             s[['SUBJECT_ID', 'HADM_ID','mode_score', 'min_score', 'max_score']], on = ['SUBJECT_ID', 'HADM_ID'])

n = n[['mode_score', 'min_score', 'max_score', 'nihss_score']]

n.to_csv(os.path.join(path,'x_validation_outcomes.csv'), index=False)

