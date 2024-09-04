
# ------------------------------------------------------
# Notes Preprocessing for the full train and test sets
# ------------------------------------------------------

# This script imports datasets with admissions and notes time stamps and
# aligns notes for each day of admission + following day

# Only code is provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta

# path = path here
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'df_nihss.csv')) # dataframe with patients admission timestamps (not provided - HPI)

# Dataframe df also contains patient demographics, which are used for statistics but not for modeling

notes = pd.read_csv(os.path.join(path,'nihss_notes.csv'), header=None) # dataframe with patients notes (not provided - HPI)

notes.columns = ['PatientID', 'PatientEncounterID', 'NoteID', 'NoteDSC',
                  'InpatientNoteTypeDSC', 'ContactDTS', 'LinesCount', 'NoteTXT']

notes = pd.merge(df, notes.drop(columns=['PatientEncounterID']).drop_duplicates(), on='PatientID')


# Day of admission + following day
notes['jc_admitdate_2'] = notes['jc_admitdate'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(days=1))

# ContactDTS corresponds to visit note time stamp
notes = notes[(notes.ContactDTS.astype('datetime64[ns]') == notes.jc_admitdate.astype('datetime64[ns]')) |
              (notes.ContactDTS.astype('datetime64[ns]') == notes.jc_admitdate_2.astype('datetime64[ns]'))]


# ------------------------------------------------------
# Group notes per order of time and note id
# ------------------------------------------------------

n = notes.sort_values(['ContactDTS',"NoteID","LinesCount"], ascending=[True,True,True])

n['Notes'] = n['NoteTXT']

#add spaces
n.Notes = ' ' + n.Notes.astype(str) + ' '

n = n.groupby(['PatientID', 'SexDSC', 'Age', 'jc_admitdate', 'gs_discdatetime', 'nihss_score', 'ContactDTS', 'InpatientNoteTypeDSC', 'NoteID']).Notes.sum().reset_index()

# Repeat removing duplicated spaces and words in a row

# Remove duplicated spaces
n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) 

# Remove duplicated words in row
n.Notes = n.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))

n.to_csv(os.path.join(path,'notes_ready.csv'), index=False) # ready for score extraction

n = n.sort_values(['ContactDTS',"NoteID"], ascending=[True,True])

n = n[['PatientID', 'SexDSC', 'Age', 'jc_admitdate', 'gs_discdatetime', 'nihss_score','NoteID','Notes']].drop_duplicates().reset_index(drop=True)


from text_preprocessing import process_text

n = process_text(n)

n.to_csv(os.path.join(path,'df_with_lemma.csv'), index=False)
 


