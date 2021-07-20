from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
ds = TabularDatasetFactory.from_delimited_files(path="https://raw.githubusercontent.com/JCForszp/nd00333-capstone/master/Datasets/heart_failure_clinical_records_dataset.csv")
df=ds.to_pandas_dataframe()

# Data columns

cols_list = list(df.columns)
y_cols=cols_list.pop()  # target label is located at the end of the list -> pop()
x_cols=cols_list
x = df[x_cols]; y = df[y_cols]

# Split data into train and test sets (80/20 basis).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

data = {"train": {"X": x_train, "y": y_train},
        "test": {"X": x_test, "y": y_test}}

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()
