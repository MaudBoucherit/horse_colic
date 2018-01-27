import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

## Import the data
train = pd.read_csv("../data/train.csv", index_col=0)
test = pd.read_csv("../data/test.csv", index_col=0)

## Get the missing data percentage
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns 
        
na_train = missing_values_table(train)
na_test = missing_values_table(test)

na_train = na_train.sort_index()
na_test = na_test.sort_index()

pd.concat([na_train, na_test], axis=1, keys=['train', 'test']).to_csv("../results/na_count.csv")


## Plot some of my variables
# Change the response's labels
train['surgical_lesion'] = train['surgical_lesion'].astype(str)
train['surgical_lesion'] = train['surgical_lesion'].replace('1', 'Yes')
train['surgical_lesion'] = train['surgical_lesion'].replace('2', 'No')

# Create the visualisation
alt.Chart(train).mark_circle().encode(x=alt.X('pulse', axis=alt.Axis(title='Pulse (beats per minute)')),
                                      y=alt.Y('protein', axis=alt.Axis(title='Protein rate (gms/dL)')),
                                      color=alt.Color('surgical_lesion:N', title='Need surgery?')).savechart('../results/EDA_plot.html')  
