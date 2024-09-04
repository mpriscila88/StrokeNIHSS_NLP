import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta

def process_text(n):

    # negation
    
    exp = ['w/out','w out','w/ out']
    
    for i in exp:
        n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(i, 'without'))
    
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('n\'t', ' not'))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('neither', 'not'))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' nor ', ' not '))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' w/', ' with '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' r/o ', ' rule out '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' h/o ', ' history of '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' a-fib ', ' afib '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' a fib ', ' afib '))
    
    ####################################
    # Apply NIHSS numbers to words
    #------------------------------
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' l ', ' left '))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' r ', ' right '))
    
    n.Notes = n.Notes.astype(str).apply(lambda x: x.replace(',',' ')) 
    
    # Remove duplicated spaces
    n.Notes = n.Notes.astype(str).apply(lambda x: " ".join(x.split())) 
    
    from NIHSS_numbers2words import n2w
    
    n = n2w(n)
    
    # ------------------------------------------------------
    # Remove numbers
    # ------------------------------------------------------
    
    r=re.compile(r'\d')
    
    n['Notes'] = n['Notes'].apply(lambda x: r.sub('', x))  
    
    # ------------------------------------------------------
    # Remove special characters
    # ------------------------------------------------------
    
    n.Notes = n.Notes.apply(lambda x: x.replace('/l',' '))
    
    n.Notes = n.Notes.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)) 
    
    n.Notes = n.Notes.apply(lambda x: x.replace('.',' ')) 
    
    n.Notes = n.Notes.astype(str).apply(lambda x: x.replace(',',' ')) 
    
    # Remove duplicated spaces
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) 
    
    # Remove duplicated words in row
    n.Notes = n.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))
    
    # ------------------------------------------------------
    # Abbreviation expansion
    # ------------------------------------------------------
     
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' l ', ' left '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' r ', ' right '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' hx ', ' history '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' dispo ', ' disposition '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' temp ', ' temperature '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' gi ', ' gastrointestinal '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' gu ', ' genitourinary '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' resp ', ' respiratory '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' eval ', ' evaluation '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' meds ', ' medications '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('medicines', 'medications'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('medicine', 'medication'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' cv ', ' cardiovascular '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' htn ', ' hypertension '))   
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' iv ', ' intravenous ')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' gen ', ' general ')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' dx ', ' diagnosis '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' pulm ', ' pulmonary '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ref ', ' refer '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' tx ', ' treatment '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' dm ', ' diabetes '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' abd ', ' abdominal '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('abdomen', 'abdominal'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ext ', ' extremities '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' extr ', ' extremities '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ecg ', ' electrocardiogram '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ekg ', ' electrocardiogram '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' echo ', ' echocardiogram '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' etoh ', ' alcohol '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' wt ', ' weight '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ng tube ', ' nasogastric tube '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ngtube ', ' nasogastric tube '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' max ', ' maximum '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' min ', ' minimum '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' neg ', ' negative '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ms ', ' mental status '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' onc ', ' oncology '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ot ', ' occupational therapy '))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' lle ', ' left lower extremity '))         
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' lue ', ' left upper extremity ')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' rue ', ' right upper extremity '))                                                                      
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' rle ', ' right lower extremity ')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' le ', ' lower extremity '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ue ', ' upper extremity '))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' lsw ', ' left sided weakness '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' rsw ', ' right sided weakness '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' lkw ', ' last known well '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ams ', ' altered mental status '))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' ddx ', ' differential diagnosis '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('cva tenderness', 'costovertebral angle tenderness'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' nad ', ' no abnormality detected no apparent distress no appreciable disease '))
    
    # ------------------------------------------------------
    # Abbreviation contraction
    # ------------------------------------------------------
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' regular rate and rhythm ', ' rrr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' regular rate and regular rhythm ', ' rrr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' regular rate regular rhythm ', ' rrr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('normal saline', 'ns'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('blood pressure', 'bp'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('systolic bp', 'sbp'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('diastolic bp', 'dbp'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('heart rate', 'hr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('respiratory rate', 'rr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('atrial fibrillation', 'afib'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('emergency department', 'ed'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('nothing by mouth', 'npo'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('outside hospital', 'osh'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('middle cerebral artery', 'mca')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('cranial nerves', 'cn')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('cranial nerve', 'cn')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('computed tomography', 'ct')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('ct angiography', 'cta'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('ct angio', 'cta'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' angio ', ' angiography '))                
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('magnetic resonance angiography', 'mra'))           
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('deep vein thrombosis', 'dvt'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('erythrocyte sedimentation rate', 'esr'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('history of present illness', 'hpi'))         
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('internal carotid artery', 'ica'))  
           
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('partial thromboplastin time', 'ptt'))                                                                      
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('transthoracic echocardiogram', 'tte'))                                                                      
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('tissue plasminogen activator', 'tpa'))                                                                                
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('room air', 'ra'))                                                                      
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('review of systems', 'ros')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('speech language pathologist', 'slp'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('primary care physician', 'pcp'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('chest radiography', 'cxr'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('diabetes mellitus', 'diabetes'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('past medical history', 'pmhx')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('physical therapist assistant', 'pta'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('range of motion', 'rom'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('shortness of breath', 'sob'))  
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('shortness breath', 'sob')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('transient ischaemic attack', 'tia')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('transient ischemic attack', 'tia')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('congestive heart failure', 'chf')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('intra arterial therapy', 'iat'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('intraarterial therapy', 'iat'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('subarachnoid hemorrhage ', 'sah'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('head of bed', 'hob'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('out of bed', 'oob'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('emergency medical services', 'ems'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('level of consciousness', 'loc'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('level consciousness', 'loc'))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('social history', 'sh'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('murmurs rubs and gallops', 'mgr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('murmurs rubs or gallops', 'mgr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('murmurs rubs gallops', 'mgr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('murmurs gallops or rubs', 'mgr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('murmurs gallops rubs', 'mgr'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' mrg ', ' mgr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('medical decision making', 'mdm'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('mean corpuscular volume', 'mcv'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('familial hypercholesterolemia', 'fh'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' xray ', ' xr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' x ray ', ' xr '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('by mouth', 'po'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('cerebral vascular accident', 'cva')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('cerebralvascular accident', 'cva')) 
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('coronary artery disease', 'cad'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('ear nose and throat', 'ent'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('ear nose throat', 'ent'))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace('head eyes ent', 'heent'))
    
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' mlarm ', ' motor left arm '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' mrarm ', ' motor right arm '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' mlleg ', ' motor left leg '))
    n.Notes = n.Notes.astype(str).str.lower().apply(lambda x: x.replace(' mrleg ', ' motor right leg '))
    
    # ------------------------------------------------------
    # Remove words
    # ------------------------------------------------------
    
    # nihss score - written different ways
    
    expressions = ['nihss stroke scale','nihss stroke score', 
                   'stroke scale','stroke score', 'nihss scale', 'nihss score',
                   'nihss total score','nih telestroke scale total','nihss',' vs ',
                   ' ec tablets', ' ec tablet', 'vital signs', 'vital sign']
    
    # vs vital signs or versus
    
    for i in expressions:
        n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x:  x.replace(i,' '))
    
    
    # stop words 
    
    from remove_stop_words import remove_stop_words_
    
    n = remove_stop_words_(n)
    
    # Repeat removing duplicated spaces and words in a row
    
    # Remove duplicated spaces
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) 
    
    # Remove duplicated words in row
    n.Notes = n.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))
    
    #------------------------------------------------------------------------
    # Stemming
    #------------------------------------------------------------------------ 
    
    import nltk
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    nltk.download('punkt')
    nltk.download('wordnet')
    
    def stemming(words):
        lemmatizer = PorterStemmer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.stem(word)
            lemmas.append(lemma)
        return lemmas
    
    def stem(df, column):
        df[column] = df[column].str.join(" ") # joining
        df[column] = df[column].str.strip()
        return df[column]
    
    n.Notes = n.Notes.str.split(" ") # splitting string (nltk.word_tokenize)
       
    n.Notes = n.Notes.apply(lambda x: stemming(x))
     
    n.Notes = stem(n, 'Notes') 
    
    # ------------------------------------------------------
    # Remove words
    # ------------------------------------------------------
    
    # Repeat removing duplicated spaces and words in a row
    
    # Remove duplicated spaces
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) 
    
    # Remove duplicated words in row
    n.Notes = n.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))

    return n

