# Driver script
# Maud Boucherit
# Jan 2018
#
# This script runs all the scripts necessary to build the report of the horse colic project
#
# usage: make all

# run from top to bottom
all: results/model

# Import the data
data/train.csv: src/data_import.py
	python src/data_import.py

# Produce some decriptive statistics
results/EDA: src/EDA.py data/train.csv
	python src/EDA.py data/train.csv data/test.csv

# Wrangle the data to use sklearn afterwards
results/data.pickle: src/data_wrangling.py data results/EDA
	python src/data_wrangling.py data/train.csv data/test.csv results/na_count.csv

# Build and fit a Logistic Regression
results/model: src/model_selec.py results/data.pickle
	python src/model_selec.py results/data.pickle

# clean up intermediate files
clean:
	rm -f data/*.csv
	rm -f results/*.csv results/*.pickle
	rm -f results/*.png results/*.html
