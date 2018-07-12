import os
import pandas as pd

import argparse
import numpy as np
from libs.utils import *
import matplotlib.pyplot as plt
import seaborn as sns


input_dir = os.path.join(os.getcwd(), 'input')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
sample_size = None
if __name__ == '__main__':
    app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv.zip'), nrows=sample_size)
    app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv.zip'), nrows=sample_size)
    bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv.zip'), nrows=sample_size)
    bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv.zip'), nrows=sample_size)
    credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv.zip'), nrows=sample_size)
    pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv.zip'), nrows=sample_size)
    prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv.zip'), nrows=sample_size)
    install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv.zip'), nrows=sample_size)
    print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
    print('Main application test data set shape = {}'.format(app_test_df.shape))
    print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))


    # Merge the datasets into a single one for training
    len_train = len(app_train_df)
    app_both = pd.concat([app_train_df, app_test_df])
    with timer('Feature engineering ...'):
        merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,
                                        pos_cash_df, prev_app_df, install_df)
    if sample_size is None:
        merged_df.to_csv(os.path.join(input_dir, 'data_feature_engineering.csv'))