#!/usr/bin/env python 
# data_wrangling.py
# Maud Boucherit, Jan 2018
#
# This script select variables, reformat categorical variable,
# impute the missing data and save a matrix version of the data
#
# Dependencies: argparse, pandas, fancyimpute, pickle
#
# Usage: python src/data_wrangling.py data/train.csv data/test.csv results/na_count.csv

# import libraries
import argparse
import pandas as pd
from fancyimpute import KNN 
import pickle

# read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_train')
parser.add_argument('input_test')
parser.add_argument('input_na')
args = parser.parse_args()

def main():
    ## Import the data
    train = pd.read_csv(args.input_train, index_col=0)
    test = pd.read_csv(args.input_test, index_col=0)
    na_count = pd.read_csv(args.input_na, index_col=0, header=[0,1])

    ## Drop columns with more than 20% of missing values
    na_col = na_count.loc[na_count['train']['pct_na'] > 20 ].index
    train = train.drop(na_col, axis = 1)
    test = test.drop(na_col, axis = 1)
    
    ## Recode binary variables
    train['age'] = train['age'].replace([1, 9], [0,1])
    train['surgical_lesion'] = train['surgical_lesion'].replace([2, 1], [0,1])
    test['age'] = test['age'].replace([1, 9], [0,1])
    test['surgical_lesion'] = test['surgical_lesion'].replace([2, 1], [0,1])
    train['capillary_time'] = train['capillary_time'].replace([1, 2, 3], [0, 1, 1])
    
    ## Arrange variables
    train = train.reindex_axis(sorted(train.columns), axis=1)
    test = test.reindex_axis(sorted(test.columns), axis=1)
    
    ## Create labels
    y = list(train['surgical_lesion'])
    yvalidate = list(test['surgical_lesion'])
    
    
    ## Impute missing values
    X = KNN(k=3).complete(train.iloc[:,:-1])
    Xvalidate = KNN(k=3).complete(test.iloc[:,:-1])
    
    ## Save as pickle
    data = {'X':X, 'y':y, 'Xvalidate':Xvalidate, 'yvalidate':yvalidate}
    pickle_out = open("results/data.pickle","wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

# call main function
if __name__ == "__main__":
    main()    