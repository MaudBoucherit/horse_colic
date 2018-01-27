import pandas as pd

# Names of my variables
names = ['surgery', 'age', 'hospital', 'rectal_temp', 'pulse', 'respiration', 'extreme_temp', 
         'peripheral_pulse', 'mucous', 'capillary_time', 'pain', 'peristalsis', 'abdominal_dist', 
         'nasogastric_tube', 'nasogastric_reflux', 'nasogastric_PH', 'feces', 'abdomen', 'cell_vol',
         'protein', 'abdomino_appearance', 'abdom_protein', 'outcome', 'surgical_lesion', 'type1', 
         'type2', 'type3', 'cp_data', 'trash'] 


# Import the training data
## There is a 29th blank column for all rows except the first one, so I need to remove it
train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data", 
                    sep=" ", header=None, names=names, skiprows=1, na_values='?', 
                    usecols=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23])

# Add the first row
train = train.append(pd.DataFrame([[1, 38.50, 66, 28, 3, 3, 2, 5, 4, 4, 3, 5, 45.00, 8.40, 2]], 
                    columns=['age', 'rectal_temp', 'pulse', 'respiration', 'extreme_temp', 
                    'peripheral_pulse', 'capillary_time', 'pain', 'peristalsis', 'abdominal_dist', 
                    'feces', 'abdomen', 'cell_vol', 'protein', 'surgical_lesion']))

train.to_csv("../data/train.csv") 


# Import the test data
test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test", 
                    sep=" ", header=None, names = names[:-1], na_values='?', 
                    usecols=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23])

test.to_csv("../data/test.csv")