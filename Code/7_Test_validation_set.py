
# ------------------------------------------------------
# Test model on the validation set
# ------------------------------------------------------

# This script:
    
# imports the validation set variables from 5_Notes_preprocessing_validation_set.py
    
# imports the validation set outcomes from 6_NIHSS_score_extraction_validation_set.py

# calculates modeling performance for the validation set

# plots performance

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


# ------------------------------------------------------
# Load the validation set
# ------------------------------------------------------

x_test = pd.read_csv(os.path.join(path,'x_validation_variables.csv'))

X_test = pd.read_csv(os.path.join(path,'x_validation_outcomes.csv'))

y_test = X_test.nihss_score

###########################
# Performance functions
###########################

# configure bootstrap

from numpy import median, percentile
from numpy.random import seed, randint

def get_CI_boot_outcome(y_true,y_pred,boot):
    # bootstrap confidence intervals
    # seed the random number generator
    seed(1)
    i = 0
    # generate dataset
    dataset = y_pred
    real = y_true
    # bootstrap
    scores = list()
    while i < boot:
        # bootstrap sample
        indices = randint(0, len(y_pred) - 1, len(y_pred))
        sample = dataset[indices]
        real = y_true[indices]
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    	# calculate and store statistic 
        else:
            statistic = mean_squared_error(real,sample, squared=False)
            scores.append(statistic)
            i += 1
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower =  np.percentile(scores, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper =  np.percentile(scores, p)
    return lower,upper
    
# Spearman correlation between target vs predicted (and confidence interval) 

from scipy import stats
import pingouin as pg


###########################
# Load and test the model
###########################


import pickle

# load the model from disk
filename = 'nihss_model_new.sav'

clf = pickle.load(open(filename, 'rb'))

# ------------------------------------------------------
# Validate model
# ------------------------------------------------------

from sklearn.metrics import mean_squared_error

y_pred = clf.predict(x_test)

y_pred[y_pred<0] = 0

y_pred[y_pred>42] = 42

c = ((X_test.mode_score == X_test.min_score) & (X_test.max_score == X_test.mode_score) & (X_test.mode_score.astype(str) != 'nan'))


##############
# stage 1 only
##############

X_test['extracted'] = 0
X_test['extracted'][c] = 1

ind = X_test[X_test['extracted']==1]

y_t=y_test.loc[ind.index]
y_p=X_test.mode_score[c]

mean_squared_error(y_t, y_p, squared=False)

lower,upper = get_CI_boot_outcome(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values,boot=1000)

print(lower, round(mean_squared_error(y_t, y_p, squared=False) , 4), upper)

# Spearman correlation between target vs predicted (and confidence interval) 

x = stats.spearmanr(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values)[0] 

stat = stats.spearmanr(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values)[0]

ci = pg.compute_bootci(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values, func='spearman', n_boot=1000, confidence=0.95,
                       paired=True, seed=42, decimals=4)

print(round(stat, 4), ci)


##############
# stage 2 only
##############

ind = X_test[X_test['extracted']==0]

y_t=y_test.loc[ind.index]
y_p=pd.DataFrame(y_pred).loc[ind.index]

mean_squared_error(y_t, y_p, squared=False) 

lower,upper = get_CI_boot_outcome(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values,boot=1000)

print(lower, round(mean_squared_error(y_t, y_p, squared=False) , 4), upper)

# Spearman correlation between target vs predicted (and confidence interval) 

x = stats.spearmanr(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values)[0] 

stat = stats.spearmanr(y_t.reset_index(drop=True).values,y_p.reset_index(drop=True).values)[0]

ci = pg.compute_bootci(y_t.reset_index(drop=True).values,y_p[0].values, func='spearman', n_boot=1000, confidence=0.95,
                       paired=True, seed=42, decimals=4)

print(round(stat, 4), ci)


##############
# 2-stage
##############

c = ((X_test.mode_score == X_test.min_score) & (X_test.max_score == X_test.mode_score) & (X_test.mode_score.astype(str) != 'nan'))

y_pred[c] = X_test.mode_score[c] # a single score was captured

mean_squared_error(y_test, y_pred, squared=False) 

lower,upper = get_CI_boot_outcome(y_test,y_pred,boot=1000)

print(lower, round(mean_squared_error(y_test, y_pred, squared=False) , 4), upper)

# Spearman correlation between target vs predicted (and confidence interval) 

x = stats.spearmanr(y_test, y_pred)[0] 

stat = stats.spearmanr(y_test, y_pred)[0]

ci = pg.compute_bootci(y_test, y_pred, func='spearman', n_boot=1000, confidence=0.95,
                       paired=True, seed=42, decimals=4)

print(round(stat, 4), ci)


#------------------------------------------------------------------------
# Target vs prediction - plot
#------------------------------------------------------------------------


from matplotlib.cm import ScalarMappable
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 22})
from operator import itemgetter


import seaborn as sns
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

rnd_state = np.random.RandomState(0)
# Adding gaussian jitter
jitter_gt = rnd_state.normal(0, 1, size=y_test.shape[0])

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5
    
plt.scatter(y_test+jitter_gt, y_pred, c='crimson', alpha=0.6, s=100)
plt.rcParams["font.family"] = "Cambria"
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Target NIHSS', fontsize=25)
plt.ylabel('Predicted NIHSS', fontsize=25)
plt.axis('equal')
plt.xlim([0,40])
plt.ylim([0,40])
# plt.xlim([0,3])
# plt.ylim([0,3])
plt.yticks(np.arange(0,46,5))
plt.ylim([0,42])
plt.show()



