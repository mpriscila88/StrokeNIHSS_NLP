
# ---------------------------------------------------------------
# Test ordinal regression model on the test and validation sets
# --------------------------------------------------------------

# This script:
    
# imports the train and validation sets with reduced set of features
    
# calculates modeling performance for both sets

# plots performance and features importance

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

from performance import perf

x_train = pd.read_csv(os.path.join(path,'xtrain_variables13.csv'))
x_test = pd.read_csv(os.path.join(path,'xtest_variables13.csv'))

X_train = pd.read_csv(os.path.join(path,'x_train_outcomes.csv'))
X_test = pd.read_csv(os.path.join(path,'x_test_outcomes.csv'))

y_train = X_train.nihss_score
y_test = X_test.nihss_score


#------------------------------------------------------------------------
# Labels
#------------------------------------------------------------------------

y_train[y_train <= 4] = 0
y_train[(y_train >= 5) & (y_train <= 15)] = 1
y_train[(y_train>= 16) & (y_train <= 20)] = 2
y_train[y_train>= 21] = 3

y_test[y_test <= 4] = 0
y_test[(y_test >= 5) & (y_test <= 15)] = 1
y_test[(y_test>= 16) & (y_test <= 20)] = 2
y_test[y_test>= 21] = 3


labels = [ 'Minor stroke', 'Moderate stroke', 'Moderate to severe stroke', 'Severe stroke']

#------------------------------------------------------------------------
# Performance
#------------------------------------------------------------------------

probs = pd.read_csv(os.path.join(path,'y_pred_test20.csv'), index_col=0)

probs = np.array(probs[['P[Y=1]', 'P[Y=2]', 'P[Y=3]', 'P[Y=4]']])

y_pred = np.argmax(probs,axis=1)


c = ((X_test.mode_score == X_test.min_score) & (X_test.max_score == X_test.mode_score) & (X_test.mode_score.astype(str) != 'nan'))


##############
# stage 1 only
##############

X_test['extracted'] = 0
X_test['extracted'][c] = 1

ind = X_test[X_test['extracted']==1]

y_t=y_test.loc[ind.index].reset_index(drop=True)
y_p=X_test.mode_score[c].reset_index(drop=True)


ps=probs[c]
ps = pd.DataFrame(ps*0, columns=['0','1','2','3'])

ps['0'][y_p <= 4] = 1
ps['1'][(y_p >= 5) & (y_p <= 15)] = 1
ps['2'][(y_p>= 16) & (y_p <= 20)] = 1
ps['3'][y_p>= 21] = 1

y_p[c & (y_p <= 4)] = 0
y_p[c & ((y_p >= 5) & (y_p <= 15))] = 1
y_p[c & ((y_p>= 16) & (y_p <= 20))] = 2
y_p[c & (y_p>= 21)] = 3

boot_all_micro, boot_all_macro, boot_label = perf(y_t, y_p, np.array(ps), labels)

##############
# stage 2 only
##############

ind = X_test[X_test['extracted']==0]

y_t = y_test.loc[ind.index]
y_p = pd.DataFrame(y_pred).loc[ind.index]

ps=probs[~c]

boot_all_micro, boot_all_macro, boot_label = perf(y_t, y_p, ps, labels) 


##############
# 2-stage
##############

y_pred[c] = X_test.mode_score[c] # a single score was captured

ps = pd.DataFrame(probs, columns=['0','1','2','3'])

ps[c] = ps[c]*0

ps['0'][c & (y_pred <= 4)] = 1
ps['1'][c & ((y_pred >= 5) & (y_pred <= 15))] = 1
ps['2'][c & ((y_pred>= 16) & (y_pred <= 20))] = 1
ps['3'][c & (y_pred>= 21)] = 1


y_pred[c & (y_pred <= 4)] = 0
y_pred[c & ((y_pred >= 5) & (y_pred <= 15))] = 1
y_pred[c & ((y_pred>= 16) & (y_pred <= 20))] = 2
y_pred[c & (y_pred>= 21)] = 3


boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, np.array(ps), labels) 


#------------------------------------------------------------------------
# Feature importance 
#------------------------------------------------------------------------

a=pd.read_csv(os.path.join(path, 'coef20.csv'), sep='\s+')
a = a.iloc[1:,:].reset_index(drop=True)

x_train = pd.read_csv(os.path.join(path,'xtrain_variables13.csv'))
c=pd.DataFrame(x_train.columns, columns=['Features'])

ca = pd.concat([a,c], axis=1)

ca= ca[~(ca['logit(P[Y=2|1<=Y<=2])']==0)].reset_index(drop=True)

#------------------------------------------------------------------------
# Number of uni, bi and trig in the set of features
#------------------------------------------------------------------------

from nltk.tokenize import word_tokenize 

ca['n'] = ca.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

len(ca[ca.n == 1]) 
len(ca[ca.n == 2]) 
len(ca[ca.n == 3]) 


#------------------------------------------------------------------------
# Feature importance estimates - plot - Not normalized
#------------------------------------------------------------------------
 
# Select N from top features 

N = 20
 
a = ca.Features
cols = ['logit(P[Y=2|1<=Y<=2])']
coef0 = ca[cols].T.reset_index(drop=True)
coef = abs(ca[cols]).T.reset_index(drop=True)
        

feature_importance = np.array(coef.T[0])
feature_signal = np.array(coef0.T[0])
sorted_idx = np.argsort(np.transpose(feature_importance))
feature_importance = 100.0 * (feature_importance / feature_importance.max())
topN = sorted_idx[-N:]


coef0.columns = ca.Features

c = ca.iloc[topN]

top_features = c[['Features']]
  
# Plot

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 22})
from operator import itemgetter

import seaborn as sns
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

def plt_importance_all(coef0, features, top_features, size):
    
    indices, L_sorted = zip(*sorted(enumerate(coef0.iloc[0,:]), key=itemgetter(1)))
    
    var = pd.DataFrame(features[pd.DataFrame(indices)[0]],columns=['Features'])
      
    # Select top features only
    var = var[var.Features.isin(top_features.Features)]    
    L_sorted = pd.DataFrame(L_sorted,columns=['coef'])
    L_sorted = L_sorted[L_sorted.index.isin(var.index)]                          
    
   
    data_x = var.Features
    data_hight = sorted(L_sorted.coef)
    
    data_hight_normalized = pd.DataFrame( [x for x in data_hight]) # not normalized
    
    
    
    my_cmap = plt.cm.get_cmap('coolwarm')
    colors = my_cmap(data_hight_normalized[0])
    
    fig = plt.figure(figsize=(20, size)) 
    ax = fig.add_subplot(1, 1, 1) 
    ax.barh(data_x, data_hight_normalized[0], color ='tab:blue') 

    plt.rcParams["font.family"] = "Cambria"
    plt.grid(color='grey', linestyle=':', linewidth=1)
    # plt.xlabel('Relative features importance (%)')

    get_indexes_neg = data_hight_normalized.apply(lambda x: x[x<0].index)
    b = ax.barh(data_x, data_hight_normalized[0], color='crimson',alpha=0.8)
    for ind in range(len(get_indexes_neg)):
        b[get_indexes_neg[0][ind]].set_color('#346cb0')
     
    ax.set_xticks([-1,-0.5,0,0.5,1])
    
    ax.set_yticklabels(data_x, fontsize=35)
    ax.set_xticklabels([-1,-0.5,0,0.5,1],fontsize=35)
    ax.set_xlabel('Feature Importance', fontsize=35)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#346cb0', label='Negative coefficient')
    blue_patch = mpatches.Patch(color='crimson', alpha=0.8, label='Positive coefficient')
    plt.rcParams["font.family"] = "Cambria"
    plt.legend(handles=[red_patch, blue_patch],loc='lower right',prop={'size': 32})
    plt.rcParams["font.family"] = "Cambria"

    
plt_importance_all(coef0, ca.Features, top_features, size=20)


#####################################
# Test the model on validation data
####################################

x_test = pd.read_csv(os.path.join(path,'xvalidation_variables13.csv'))

X_test = pd.read_csv(os.path.join(path,'x_validation_outcomes.csv'))

y_test = X_test.nihss_score

y_test[y_test <= 4] = 0
y_test[(y_test >= 5) & (y_test <= 15)] = 1
y_test[(y_test>= 16) & (y_test <= 20)] = 2
y_test[y_test>= 21] = 3

probs = pd.read_csv(os.path.join(path,'y_pred_validation20.csv'), index_col=0)

probs = np.array(probs[['P[Y=1]', 'P[Y=2]', 'P[Y=3]', 'P[Y=4]']])

y_pred = np.argmax(probs,axis=1)

c = ((X_test.mode_score == X_test.min_score) & (X_test.max_score == X_test.mode_score) & (X_test.mode_score.astype(str) != 'nan'))



X_test['extracted'] = 0
X_test['extracted'][c] = 1

##############
# stage 1 only
##############

ind = X_test[X_test['extracted']==1]

y_t=y_test.loc[ind.index].reset_index(drop=True)
y_p=X_test.mode_score[c].reset_index(drop=True)


ps=probs[c]
ps = pd.DataFrame(ps*0, columns=['0','1','2','3'])

ps['0'][y_p <= 4] = 1
ps['1'][(y_p >= 5) & (y_p <= 15)] = 1
ps['2'][(y_p>= 16) & (y_p <= 20)] = 1
ps['3'][y_p>= 21] = 1

y_p[c & (y_p <= 4)] = 0
y_p[c & ((y_p >= 5) & (y_p <= 15))] = 1
y_p[c & ((y_p>= 16) & (y_p <= 20))] = 2
y_p[c & (y_p>= 21)] = 3


boot_all_micro, boot_all_macro, boot_label = perf(y_t, y_p, np.array(ps), labels) 

##############
# stage 2 only
##############

ind = X_test[X_test['extracted']==0]

y_t=y_test.loc[ind.index]
y_p=pd.DataFrame(y_pred).loc[ind.index]
ps=probs[~c]

boot_all_micro, boot_all_macro, boot_label = perf(y_t, y_p, ps, labels) 


##############
# 2-stage
##############

y_pred[c] = X_test.mode_score[c] # a single score was captured

ps = pd.DataFrame(probs, columns=['0','1','2','3'])

ps[c] = ps[c]*0

ps['0'][c & (y_pred <= 4)] = 1
ps['1'][c & ((y_pred >= 5) & (y_pred <= 15))] = 1
ps['2'][c & ((y_pred>= 16) & (y_pred <= 20))] = 1
ps['3'][c & (y_pred>= 21)] = 1

y_pred[c & (y_pred <= 4)] = 0
y_pred[c & ((y_pred >= 5) & (y_pred <= 15))] = 1
y_pred[c & ((y_pred>= 16) & (y_pred <= 20))] = 2
y_pred[c & (y_pred>= 21)] = 3

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, np.array(ps), labels) 
