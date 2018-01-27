import pandas as pd
from fancyimpute import KNN 
from sklearn.preprocessing import OneHotEncoder
import pickle

## Import the data
train = pd.read_csv("../data/train.csv", index_col=0)
test = pd.read_csv("../data/test.csv", index_col=0)
na_count = pd.read_csv("../results/na_count.csv", index_col=0, header=[0,1])

## Drop columns with more than 20% of missing values
na_col = na_count.loc[na_count['train']['pct_na'] > 20 ].index
train = train.drop(na_col, axis = 1)
test = test.drop(na_col, axis = 1)

## Recode binary variables
train['age'] = train['age'].replace([1, 9], [0,1])
train['surgical_lesion'] = train['surgical_lesion'].replace([2, 1], [0,1])
test['age'] = test['age'].replace([1, 9], [0,1])
test['surgical_lesion'] = test['surgical_lesion'].replace([2, 1], [0,1])

## Arrange variables
train = train.reindex_axis(sorted(train.columns), axis=1)
test = test.reindex_axis(sorted(test.columns), axis=1)

## Create labels
y = list(train['surgical_lesion'])
yvalidate = list(test['surgical_lesion'])


## Impute missing values
X = KNN(k=3).complete(train.iloc[:,:-1])
Xvalidate = KNN(k=3).complete(test.iloc[:,:-1])

## Create dummies for categorical variables
onehotencoder = OneHotEncoder(categorical_features = [2,4,5])
X = onehotencoder.fit_transform(X).toarray()
Xvalidate = onehotencoder.fit_transform(Xvalidate).toarray()

## Save as pickle
data = {'X':X, 'y':y, 'Xvalidate':Xvalidate, 'yvalidate':yvalidate}
pickle_out = open("../results/data.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()
