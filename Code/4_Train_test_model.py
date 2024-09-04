
####################################
# Load the data and train the model
####################################

# This script: 

# imports the model train set from 2_Train_test_split.py to create the linear model

# imports the test set from 3_NIHSS_score_extraction_test_set.py to test the model

# calculates modeling performance for the test set

# plots performance and model features importance estimates

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

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


x_train = pd.read_csv(os.path.join(path,'x_train_variables.csv'))
x_test = pd.read_csv(os.path.join(path,'x_test_variables.csv'))


X_train = pd.read_csv(os.path.join(path,'x_train_outcomes.csv'))
X_test = pd.read_csv(os.path.join(path,'x_test_outcomes.csv'))

y_train = X_train.nihss_score
y_test = X_test.nihss_score


#------------------------------------------------------------------------
#  Modeling - Lasso
#------------------------------------------------------------------------

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

lasso = Lasso(random_state=0, max_iter=100)
alphas = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 1.5, 2, 5]

tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(x_train, y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

# Alpha parameter
scores_ = pd.DataFrame(scores)
maxpos = scores_.index[scores_[0]==max(scores_[0])] 

best_alpha = alphas[maxpos[0]] # 0.04

# Plot cv scores

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

std_error = scores_std / np.sqrt(n_folds)
plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")
# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
plt.ylabel("CV score +/- std error")
plt.xlabel("alpha")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([alphas[0], alphas[-1]])

clf = Lasso(alpha=best_alpha)
clf.fit(x_train, y_train)

# import pickle

# # save the model to disk
# filename = 'nihss_model_new.sav'
# pickle.dump(clf, open(filename, 'wb'))
  
# # load the model from disk
# clf = pickle.load(open(filename, 'rb'))

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


#################################
# Test the model on the test set
################################

y_pred = clf.predict(x_test)

y_pred[y_pred<0] = 0

c = ((X_test.mode_score == X_test.min_score) & (X_test.max_score == X_test.mode_score) & (X_test.mode_score.astype(str) != 'nan'))


##############
# stage 1 only
##############

X_test['extracted'] = 0
X_test['extracted'][c] = 1

ind = X_test[X_test['extracted']==1]

y_t = y_test.loc[ind.index]
y_p = X_test.mode_score[c]

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

y_t = y_test.loc[ind.index]
y_p = pd.DataFrame(y_pred).loc[ind.index]

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
# Top features
#------------------------------------------------------------------------

coef = pd.DataFrame(clf.coef_).T

coef.columns = x_train.columns

ind = (coef == 0).all()

ind = pd.DataFrame(ind,columns={'Bool'})

top_features = pd.DataFrame(ind.loc[(ind.Bool==False)].index, columns={'Features'})

#------------------------------------------------------------------------
# Number of uni, bi and trig in the set of features
#------------------------------------------------------------------------

from nltk.tokenize import word_tokenize 

top_features['n'] = top_features.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

print(len(top_features[top_features.n == 1]))
print(len(top_features[top_features.n == 2]))
print(len(top_features[top_features.n == 3])) 

#------------------------------------------------------------------------ 
# Select N from top features 
#------------------------------------------------------------------------

N = 20

coef = pd.DataFrame(clf.coef_, columns=['Coef'])

a = pd.DataFrame(x_train.columns, columns=['Features'])

coef = pd.concat([a, coef], axis=1)

coef = coef[coef.Features.isin(top_features.Features)]

coef.Coef = np.abs(coef.Coef)

sorted_idx = coef.Coef.sort_values(ascending=False)

sorted_idx = sorted_idx[0:N]

coef=coef[coef.index.isin(sorted_idx.index)]

top_features = coef[['Features']]
  
#------------------------------------------------------------------------
# Feature importance estimates - plot
#------------------------------------------------------------------------

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
     
    ax.set_xticks([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5])
    
    ax.set_yticklabels(data_x, fontsize=35)
    ax.set_xticklabels([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5],fontsize=35)
    ax.set_xlabel('Feature Importance', fontsize=35)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#346cb0', label='Negative coefficient')
    blue_patch = mpatches.Patch(color='crimson', alpha=0.8, label='Positive coefficient')
    plt.rcParams["font.family"] = "Cambria"
    plt.legend(handles=[red_patch, blue_patch],loc='lower right',prop={'size': 32})
    plt.rcParams["font.family"] = "Cambria"

    
coef = pd.DataFrame(clf.coef_).T

coef.columns = x_train.columns

plt_importance_all(coef, x_train.columns, top_features, size=20)


#------------------------------------------------------------------------
# Target vs prediction - plot
#------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

rnd_state = np.random.RandomState(0)
# Adding gaussian jitter
jitter_gt = rnd_state.normal(0, 1, size=y_test.shape[0])

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
plt.rcParams["font.family"] = "Cambria"
plt.axis('equal')
plt.xlim([0,40])
plt.ylim([0,40])
# plt.xlim([0,3])
# plt.ylim([0,3])
plt.rcParams["font.family"] = "Cambria"
plt.show()


