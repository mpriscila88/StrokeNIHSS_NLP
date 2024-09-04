import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta


def extract_scores(n):
    
    n.Notes = n.Notes.astype(str).apply(lambda x: " ".join(x.split())) # removes duplicated spaces
    
    n.Notes = n.Notes.astype(str).str.lower()
   
    # remove all dates
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('nihss date:?\s?([\w]).{1,25} score','nihss',x)) #
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('nihss time of exam:? \d\d?:?.?\d\d?\s?(am)?(pm)?\s?(total)?\s?score','nihss',x)) #
     
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('on \d\d?\/?\d\d? am',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('on \d\d?\/?\d\d? pm',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('at \d\d?\/?\d\d? am',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('at \d\d?\/?\d\d? pm',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date:?\s?\d\d?\/\d\d?,?\s? time:?\s?\d\d?.?\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\/\d\d?\/\d\d\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?-\d\d?-\d\d\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d\s?\d\d\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date time',' ',x))
   
   
    # numbers in extense to number
   
    n['Notes'] = n['Notes'].astype(str).str.replace(' zero',' 0')
    n['Notes'] = n['Notes'].astype(str).str.replace(' one',' 1')
    n['Notes'] = n['Notes'].astype(str).str.replace(' two',' 2')
    n['Notes'] = n['Notes'].astype(str).str.replace(' three',' 3')
    n['Notes'] = n['Notes'].astype(str).str.replace(' four',' 4')
    n['Notes'] = n['Notes'].astype(str).str.replace(' five','5')
    n['Notes'] = n['Notes'].astype(str).str.replace(' six',' 6')
    n['Notes'] = n['Notes'].astype(str).str.replace(' seven',' 7')
    n['Notes'] = n['Notes'].astype(str).str.replace(' eight',' 8')
    n['Notes'] = n['Notes'].astype(str).str.replace(' nine',' 9')
    n['Notes'] = n['Notes'].astype(str).str.replace(' ten',' 10')
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub(' 0\d\d?\d?\d?',' ',x)) 
                                              
    # remove numbers > 42
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('4[3-9]|[5-9][0-9]|[1-9]\d{2,}',' ',x))
   
    # 4[3-9] matches from 43-59
    # [5-9][0-9] matches from 50-99
    # [1-9]\d{2,} matches > 100
   
     
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('hospital\d?\s?\d\d?',' ',x))
   
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\[\*\*\s?([\w]).{1,4}\s?\*\*\]',' ',x))
   
   
   
    # remove uncertain numbers
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nihss <',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nihss >',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('<=?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('>=?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\s?hour',' ',x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\s?hr',' ',x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('values 0 \d',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\s?point',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\(\d',' ',x)) # subcomponents
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub(r'[nihss:]\d/\d/\d/\d/\d/\d/\d/\d/\d/\d/\d/\d/\d/\d/\d=', ' ', x)) # subcomponents
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub(r'nih(ss)?\s?(stroke)?\s?(scale)?\s?(score)?\s?total',' nihss ', x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub(r'nih\s?([\w]).{1,100} total:',' nihss ', x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub(r'nih\s?([\w]).{1,100} total\s?=',' nihss ', x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nihss: ([\w]).{1,35} score =','nihss ', x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nihss:? ([\w]).{1,70} total score','nihss ', x))  ##################################
   
    n.Notes = n.Notes.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)) # removes special characters
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nih\s?([\w]).{1,25} total (was)?(is)?(equals)?','nihss ', x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('nih\s?([\w]).{1,25} score (was)?(is)?(equals)?','nihss ', x)) 
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d? \d\d? \d\d\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d? d\d? \d\d\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d \d\d\ \d\d',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?\s?\d\d\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d\d\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d\d',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('tpa initial \d\d? \d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d? \d\d? time score',' ',x))
   
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces
   
    # Repeat with no spaces
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?\s?\d\d\d?\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d?\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d\d\s?\d\d?',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d\d',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('time \d',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date (time)?\s?\d\d \d\d?',' ',x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date (time)?\d\d \d\d?',' ',x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date (time)?\s?\d \d\d?',' ',x)) 
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date (time)?\d \d\d?',' ',x)) 
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('date \d\d? time \d\d? \d\d?',' ',x)) 
   
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?.?\d\d\s?am',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?.?\d\d\s?pm',' ',x))
   
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\s?am',' ',x))
    n['Notes'] = n['Notes'].astype(str).apply(lambda x: re.sub('\d\d?\s?pm',' ',x))
   
   
    # ------------------------------------------------------
    # nihss score - written different ways
    # ------------------------------------------------------
   
    expressions = ['total nihss score', 'nihss total score', 'total nihss', 
                   'nihss total', 'nihss stroke scale','nihss stroke score', 'nihss scale',
                   'nihss score', 'nihss total score', 'nihss score total'
                   'nih telestroke scale total','nih ss', 'nih stroke scale',
                   'nih stroke score', 'nih stroke scale score', 
                   'stroke scale','stroke score', 'nihss of', 'nihss=', 
                   'nih telestroke scale score', 'nih']
   
    for i in expressions:
        n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x:  x.replace(i,' nihss ')) # spaces
        
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('[a-z]\d\d?',' ',x)) # here
   
                
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub(r'\.+','.',x)) #     
                        
    # ------------------------------------------------------
    # Extract nihss score
    # ------------------------------------------------------
   
    #Find expression
    def find_(s, d, ntokens):     
        s = pd.Series(s).str.extractall('('+ d + '(:?\??[?)(\s?)([\w]).{1,'+ntokens+'})').iloc[:,0].str.cat(sep=' -#- ')
        return s
   
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces
     
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('nihss date time score','nihss',x)) #
     
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('nihss closest to ia procedure at this hospital','nihss ia procedure',x)) # 
        
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('nihss\s?(ss)?\s?closest to ia procedure','nihss ia procedure',x)) # 
     
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction \d \d \d','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction \d \d','extinction',x)) #
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction \d','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction inattention \d (total)?','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction and inattention \d (total)?','extinction',x)) #
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('inattention formerly neglect \d','extinction',x)) # 
     
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('formerly neglect \d','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('neglect\s?([\w]).{1,35} score','nihss',x)) #
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction total score','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('extinction and extinction','extinction',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('distal motor function \d','distal motor function',x)) # 
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('distal motor function normal total score','motor function total',x)) #
   
    n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub('distal motor function normal total','motor function total',x)) #
   
        
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces
   
    exps = ['nihss date time','nihss date','nihss time']
   
    for exp in exps:
        n['Notes'] = n['Notes'].astype(str).str.lower().apply(lambda x: re.sub(exp,' ',x)) # 
   
    n['nihss'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'nihss',ntokens='40'))
   
    n['nihss'] = n['nihss'].apply(lambda x: " ".join(x.split())) # removes duplicated spaces
   
   
    n['total 1'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'abnormality total',ntokens='3'))
   
    n['total 2'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'extinction',ntokens='10'))
   
    n['total 3'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'modalities total',ntokens='3'))
   
    n['total 4'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'modality total',ntokens='3'))
   
    n['total 5'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'motor function total',ntokens='3'))
   
    n['total 6'] =  pd.Series(n['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'dysarthria total',ntokens='3'))
   
   
    n["nihss"] = n["nihss"] + ' -#- ' + n['total 1'] + ' -#- ' + n['total 2'] + ' -#- ' + n['total 3'] + ' -#- ' + n['total 4'] + ' -#- ' + n['total 5'] + ' -#- ' + n['total 6']
   
    n['nihss'] = n['nihss'].apply(lambda x: " ".join(x.split())) # removes duplicated spaces
   
    s = n["nihss"].str.split(' -#- ', expand=True)
   
    expressions = ["\d\d\s?.?\d\d am", "\d\d\s?.?\d\d pm", "\d\s?.?\d\d am", "\d\s?.?\d\d pm", "\d?\d\s?am", "\d?\d\s?pm", 
                   "\d\d?\s?mg", "bp.*", "meds.*", "bed.*", "\d\d?d.*", "range.*", "labs.*", "photos.*",  "phone number.*", 
                   "temperature.*", "glucose.*", "\d\d\s?y", "\d\d f?e?male", "no data found.*", "gcs.*", "mrs.*", 
                   "month \d\d?",  "  day \d\d?",
                   "\d\d?\s?days?", "\d\d\s?min", "1a.*", "1b.*", "1 loc.*", "\d\d?\.\d\d?", "\d\d?\:\d\d?"]
   
   
    for i in range(0,s.shape[1]):
        s[i] = s[i].astype(str)
        for k in expressions:
            s[i] = s[i].apply(lambda x:  re.sub(k," ",x))        
   
    for i in range(0,s.shape[1]):    
        m = s[i].astype(str).apply(lambda x:  re.search(r'\d+', x))
        s[i][m.astype(str) =='None'] = np.nan
        s[i][m.astype(str) !='None'] = m[m.astype(str) !='None'].apply(lambda x: x.group(0))
   
    s['mode_score'] = np.nanmax(s.astype(float).mode(axis='columns',numeric_only=True), axis = 1)
    s['min_score'] = np.nanmin(s.astype(float), axis = 1)
    s['max_score'] = np.nanmax(s.astype(float), axis = 1)
      
    s2 = pd.concat([n, s], axis = 1)
    
    return s2