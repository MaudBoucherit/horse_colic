# Horse Colic - Source
This folder contains the scripts for the Horse Colic.

Maud Boucherit, Jan 2018

### data_import.py
This script import the data for the horse colic project and deal with the missing data before saving it in [data/](data/).

Dependencies: `argparse`, `pandas`

Usage: `python src/data_import.py`


### EDA.py
This script create some exploratory descriptive statistics about the data, like the repartition of missing values or a visualisation.

Dependencies: `argparse`, `pandas`, `matplotlib.pyplot`, `altair`

Usage: `python src/EDA.py data/train.csv data/test.csv`


### data_wrangling.py
This script select variables, reformat categorical variable, impute the missing data and save a matrix version of the data.

Dependencies: `argparse`, `pandas`, `fancyimpute`, `pickle`

Usage: `python src/data_wrangling.py data/train.csv data/test.csv results/na_count.csv`


### model_selec.py
This script build and fit a Logistic regression to the training data with random features elimination.

Dependencies: `argparse`, `pickle`, `numpy`, `pandas`, `matplotlib.pyplot`, `sklearn`

Usage: `python src/model_selec.py results/data.pickle`
