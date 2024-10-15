### FOR TESTING CORRELATION

import pandas as pd

data_file = 'denmarkgrid.csv'
data_df = pd.read_csv(data_file)

data_df = data_df.drop("soilType", axis=1)

data_df['positiveOccurences'] = data_df['positiveOccurences'].apply(lambda x: 1 if x > 0 else x)

correlation_matrix = data_df.corr('spearman') # 'pearson' 'kendall' 'spearman'
print(correlation_matrix["positiveOccurences"])