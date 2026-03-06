import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

df = pd.read_csv('data/raw_data/raw.csv')

X = df.drop(['date','silica_concentrate'], axis = 1)
y = df.silica_concentrate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

save_dataframes(X_train, X_test, y_train, y_test, 'data/processed_data' )

