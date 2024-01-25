import json
import pandas as pd
import pprint, logging
import seaborn as sns
import matplotlib.pyplot as plt


#plotting modules
from plotting import violin_plot, corr_matrix, KDE_scatter_plot

from logistic_regression import log_reg_probability


# extract json to data
with open('output.json', 'r') as file:
    data = json.load(file)

#print(len(data.items()))

# Flattening the data and extracting relevant parts
flattened_data = []
for title, content in data.items():
    for key in ['positive', 'negative']:
        entry = content[key]['data']
        entry['topic'] = title
        entry['argument_type'] = key
        #pprint.pprint(entry)
        flattened_data.append(entry)

# Creating a DataFrame
df = pd.DataFrame(flattened_data)

#calculates mean by conlumn; also cleans up the data
def calc_mean(df):
    for label in df.columns:
        if label in ('argument_type', 'topic', 'justification'):
            pass
        else:
            if df[label].dtype == 'object':
                try:
                    #some of gpt's output is a little messed up so we replace the string explanations with NaN
                    df[label] = pd.to_numeric(df[label],errors='coerce')
                except Exception as e:
                    logging.error(f"Error in converting {label} to numbers: {e}")
            #perform operation on original object; replace NaN with mean value
            df.fillna({label: df[label].mean()}, inplace=True)
            if pd.api.types.is_numeric_dtype(df[label]):
                mean_value = df[label].mean()
                mean_value_by_type = df.groupby('argument_type')[label].mean()
                median_value = df[label].median()
                median_value_by_type = df.groupby('argument_type')[label].median()
                '''
                print(f"{label}:")
                print(f"mean: {mean_value}")
                print(f"   {mean_value_by_type}")
                print(f"median: {median_value}")
                print(f"   {median_value_by_type}")
                print("------------")
                '''

calc_mean(df)

y_prob = log_reg_probability(df)
KDE_scatter_plot(df, y_prob)