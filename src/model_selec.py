#!/usr/bin/env python 
# model_selec.py
# Maud Boucherit, Jan 2018
#
# This script build and fit a Logistic regression to the training data
# with random features elimination.
#
# Dependencies: argparse, pickle, numpy, pandas, matplotlib.pyplot, 
#               sklearn
#
# Usage: python src/model_selec.py results/data.pickle 

# import libraries
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score

# read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_data')
args = parser.parse_args()

def main():
    ## Import data
    data = pd.read_pickle(args.input_data)
    X = data['X']
    y = data['y']
    Xvalidate = data['Xvalidate']
    yvalidate = data['yvalidate']
    
    
    ## Model and features selection
    # Shuffle the training data before cross-validation 
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0)
    
    # Initialise parameters and score matrix
    Cs = [1e-4, 1e-3, .01, .1, 1, 10, 100, 1e3, 1e4]
    nbs = range(4, 13)
    scores = np.zeros((9,9))
    
    for i in range(9):
        for j in range(9):
            # The model to test
            rfe = RFE(estimator = LogisticRegression(penalty='l1', C=Cs[i]), n_features_to_select=nbs[j])
        
            # Cross validate
            scores[i][j] = np.mean(cross_val_score(rfe, X, y, cv=6))
            
    # Get the best parameters
    ind_best = np.argmax(scores)
    C = Cs[ind_best // 9]
    nb = nbs[ind_best % 9]
    
    
    ## Create and save a rank matrix for the variables
    ranks = np.zeros((12,12))
    for i in range(12):
        rfe = RFE(estimator = LogisticRegression(penalty='l1', C=C), n_features_to_select=i+1)
        rfe.fit(X,y)
        ranks[i,:] = rfe.ranking_
    
    ranks = pd.DataFrame(ranks).rename(columns={0:'abdominal_dist', 1:'age', 2:'capillary_time', 3:'cell_vol', 
                                       4:'extreme_temp', 5:'mucous', 6:'pain', 7:'peristalsis', 8:'protein',
                                       9:'pulse', 10:'rectal_temp', 11:'respiration'})
    
    
    
    ## Fit the best model on the train set
    rfe = RFE(estimator = LogisticRegression(penalty='l1', C=C), n_features_to_select=nb)
    rfe.fit(X, y)
    
    # Save best parameters and associate scores in a data frame
    res = pd.DataFrame({'C':C, 'nb_feat':nb, 'test_score_max':np.max(scores), 'best_mod_train':rfe.score(X, y), 'best_mod_test':rfe.score(Xvalidate, yvalidate)}, index = [0])
    res.to_csv("results/params.csv")
    
    
    ## Get the coefficients from the best model
    ranks = ranks.append(pd.DataFrame(rfe.estimator_.coef_, columns=['abdominal_dist', 'cell_vol', 'pain', 'peristalsis', 'rectal_temp', 'respiration']), ignore_index=True)
    
    ranks = ranks.iloc[[0, 12]].rename({0:'rank', 12:'coef'})
    ranks = ranks.reindex_axis(ranks.columns[[6, 0, 7, 10, 11, 3, 9, 8, 5, 4, 2, 1]], axis=1)
    ranks.to_csv("results/ranks.csv")
    
    
    ## Color map of the score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(scores)
    fig.colorbar(cax)
    plt.xticks(range(9), ['4', '5', '6', '7', '8', '9', '10', '11', '12'])
    plt.yticks(range(9), ['1e-4', '1e-3', '1e-2', '0.1', '1', '10', '100', '1e3', '1e4'])
    plt.xlabel("number of features")
    ax.xaxis.set_label_position('top') 
    plt.ylabel("value of C")
    plt.savefig("results/scores.png")
    
# call main function
if __name__ == "__main__":
    main()