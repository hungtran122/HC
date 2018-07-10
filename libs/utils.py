import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import warnings
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder_2(df):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique()) == 2]
    # cat_cols = list(set(cat_cols + ['FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18']))
    for c in cat_cols:
        df[c] = df[c].astype(str)
        le = preprocessing.LabelEncoder()
        allvalues = df[c].unique().tolist()
        le.fit(allvalues)
        df[c] = le.transform(df[c].values)
    return df, cat_cols


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('./input/application_train.csv.zip', nrows=num_rows)
    test_df = pd.read_csv('./input/application_test.csv.zip', nrows=num_rows)
    if num_rows is None:
        tr_size = len(df)
    else:
        tr_size = num_rows
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    # df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7',
                 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_15',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    df = df.drop(dropcolum, axis=1)

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder_2(df)

    del test_df
    gc.collect()
    # return df[:tr_size], df[tr_size:], cat_cols
    return df, tr_size, cat_cols


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('./input/bureau.csv.zip', nrows=num_rows)
    bb = pd.read_csv('./input/bureau_balance.csv.zip', nrows=num_rows)
    bb, bb_cat = process_dataframe_ori(bb)
    bureau, bureau_cat = process_dataframe_ori(bureau)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    # Bureau: Closed credits - using only numerical aggregations
    return bureau_agg, bb_cat


# Preprocess previous_applications.csv
def previous_applications(num_rows=None):
    prev = pd.read_csv('./input/previous_application.csv.zip', nrows=num_rows)
    prev, cat_cols = process_dataframe_ori(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    return prev_agg, cat_cols


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('./input/POS_CASH_balance.csv.zip', nrows=num_rows)
    pos, cat_cols = process_dataframe_ori(pos)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg, cat_cols


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('./input/installments_payments.csv.zip', nrows=num_rows)
    ins, cat_cols = process_dataframe_ori(ins)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg, cat_cols


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('./input/credit_card_balance.csv.zip', nrows=num_rows)
    cc, cat_cols = process_dataframe_ori(cc)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg, cat_cols

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def merge_prev(app_data, prev_app_df):
    # Previous applications
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum', 'AMT_ANNUITY': 'sum', 'AMT_APPLICATION': 'sum',
                'AMT_DOWN_PAYMENT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT', 'TOTAL_PREV_AMT_ANNUITY',
                         'TOTAL_PREV_AMT_APPLICATION', 'TOTAL_AMT_DOWN_PAYMENT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

    # Average the rest of the previous app data
    prev_apps_avg = prev_app_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PAVG'])
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    # Previous app categorical features
    prev_app_df, cat_feats = process_dataframe_ori(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR') \
        .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    return merged_df

def merge_cc(merged_df, credit_card_df):
    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1 / credit_card_df.loc[x.index, 'MONTHS_BALANCE'])

    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CCAVG'])

    agg_funs = {'AMT_BALANCE':['mean', 'sum'],
                'AMT_CREDIT_LIMIT_ACTUAL': ['mean','sum'],
                'AMT_DRAWINGS_CURRENT': ['mean','sum'],
                'AMT_DRAWINGS_ATM_CURRENT': ['mean','sum'],
                'AMT_DRAWINGS_OTHER_CURRENT': ['mean','sum'],
                'AMT_DRAWINGS_POS_CURRENT': ['mean','sum'],
                'AMT_INST_MIN_REGULARITY': ['mean','sum'],
                'AMT_PAYMENT_CURRENT': ['mean','sum'],
                'AMT_PAYMENT_TOTAL_CURRENT': ['mean','sum'],
                'AMT_RECEIVABLE_PRINCIPAL': ['mean','sum'],
                'AMT_RECIVABLE': ['mean','sum'],
                'AMT_TOTAL_RECEIVABLE': ['mean','sum'],
                'CNT_DRAWINGS_ATM_CURRENT': ['mean','sum'],
                'CNT_DRAWINGS_CURRENT': ['mean','sum'],
                'CNT_DRAWINGS_OTHER_CURRENT': ['mean','sum'],
                'CNT_DRAWINGS_POS_CURRENT': ['mean','sum'],
                'CNT_INSTALMENT_MATURE_CUM': ['mean','sum'],
                'SK_DPD': ['mean','sum'],
                'SK_DPD_DEF': ['mean','sum'],
                }

    cc_agg = credit_card_df.groupby('SK_ID_CURR').agg(agg_funs)
    cc_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    merged_df = merged_df.merge(cc_agg, left_on='SK_ID_CURR', right_index=True, how='left')

    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist() + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left', suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    return merged_df
def merge_bureau(merged_df, bureau_df, bureau_balance_df):
    # Credit bureau data - numerical features
    credit_bureau_avgs = bureau_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])

    agg_funs = {
        'CREDIT_DAY_OVERDUE': ['std', 'mean', 'sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['std', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['std', 'mean', 'sum'],
        'AMT_CREDIT_SUM': ['std', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['std', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['std', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['std', 'mean', 'sum'],
        'AMT_ANNUITY': ['std', 'mean', 'sum'],
    }
    br_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_funs)
    br_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in br_agg.columns.tolist()])
    print(br_agg.columns)
    merged_df = merged_df.merge(br_agg, left_on='SK_ID_CURR', right_index=True, how='left')


    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                                how='left', suffixes=['', '_B_B'])
    print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))
    return merged_df
def merge_pos(merged_df, pos_cash_df):
    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1 / pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF': wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])

    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist() + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left', suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))
    return merged_df
def merge_ins(merged_df, install_df):
    # Installments data
    ins_avg = install_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_IAVG'])
    print('Shape after merging with installments data = {}'.format(merged_df.shape))
    return merged_df
def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
    """
    Process the input dataframes into a single one containing all the features. Requires
    a lot of aggregating of the supplementary datasets such that they have an entry per
    customer.

    Also, add any new features created from the existing ones
    """

    # # Add new features

    # Amount loaned relative to salary
    app_data['NEW_CREDIT_TO_ANNUITY_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']
    app_data['NEW_CREDIT_TO_GOODS_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_GOODS_PRICE']
    # app_data['NEW_DOC_IND_KURT'] = app_data[docs].kurtosis(axis=1)
    # app_data['NEW_LIVE_IND_SUM'] = app_data[live].sum(axis=1)
    app_data['NEW_INC_PER_CHLD'] = app_data['AMT_INCOME_TOTAL'] / (1 + app_data['CNT_CHILDREN'])
    # app_data['NEW_INC_BY_ORG'] = app_data['ORGANIZATION_TYPE'].map(inc_by_org)
    app_data['NEW_EMPLOY_TO_BIRTH_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['NEW_ANNUITY_TO_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / (1 + app_data['AMT_INCOME_TOTAL'])
    app_data['NEW_SOURCES_PROD'] = app_data['EXT_SOURCE_1'] * app_data['EXT_SOURCE_2'] * app_data['EXT_SOURCE_3']
    app_data['NEW_EXT_SOURCES_MEAN'] = app_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    app_data['NEW_SCORES_STD'] = app_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    app_data['NEW_SCORES_STD'] = app_data['NEW_SCORES_STD'].fillna(app_data['NEW_SCORES_STD'].mean())
    app_data['NEW_CAR_TO_BIRTH_RATIO'] = app_data['OWN_CAR_AGE'] / app_data['DAYS_BIRTH']
    app_data['NEW_CAR_TO_EMPLOY_RATIO'] = app_data['OWN_CAR_AGE'] / app_data['DAYS_EMPLOYED']
    app_data['NEW_PHONE_TO_BIRTH_RATIO'] = app_data['DAYS_LAST_PHONE_CHANGE'] / app_data['DAYS_BIRTH']
    app_data['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = app_data['DAYS_LAST_PHONE_CHANGE'] / app_data['DAYS_EMPLOYED']
    app_data['NEW_CREDIT_TO_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']

    # # Aggregate and merge supplementary datasets
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    merged_df = merge_prev(app_data, prev_app_df)
    merged_df = merge_cc(merged_df, credit_card_df)
    merged_df = merge_bureau(merged_df, bureau_df, bureau_balance_df)
    merged_df = merge_pos(merged_df, pos_cash_df)
    merged_df = merge_ins(merged_df, install_df)

    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    print('Shape after merging with counts data = {}'.format(merged_df.shape))

    return merged_df
def process_dataframe_2(input_df, len_train):
    """ Process a dataframe into a form useable by LightGBM """
    # Capture other categorical features not as object data types:


    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()

        # input_df[:len_train][feat] = encoder.fit_transform(input_df[:len_train][feat].fillna('missing')).astype('int32')
        # input_df[len_train:][feat] = encoder.transform(input_df[len_train:][feat].fillna('missing')).astype('int32')
        # input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        input_df[feat] = input_df[feat].astype(str)
        allvalues = input_df[feat].unique().tolist()
        encoder.fit(allvalues)
        input_df[feat] = encoder.transform(input_df[feat].values)

        if input_df[feat].isnull().sum() > 0:
            print('{} has {} null values'.format(feat, input_df[feat].isnull().sum()))
            n_values = len(input_df[feat].unique())
            input_df[feat].fillna(n_values + 10, inplace=True)
        # print(feat, input_df[feat].isnull().sum(), input_df[feat].dtypes)
        # input_df[feat] = input_df[feat].astype('int64')
        # print(feat, input_df[feat].dtypes)
        encoder_dict[feat] = encoder

    return input_df, categorical_feats.tolist(), encoder_dict
def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """
    non_obj_categoricals = [
        'FONDKAPREMONT_MODE',
        'HOUR_APPR_PROCESS_START',
        'HOUSETYPE_MODE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'WALLSMATERIAL_MODE',
        'WEEKDAY_APPR_PROCESS_START',
        'NAME_CONTRACT_TYPE_BAVG',
        'WEEKDAY_APPR_PROCESS_START_BAVG',
        'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS',
        'NAME_PAYMENT_TYPE',
        'CODE_REJECT_REASON',
        'NAME_TYPE_SUITE_BAVG',
        'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO',
        'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE',
        'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP',
        'PRODUCT_COMBINATION',
        'NAME_CONTRACT_STATUS_CCAVG',
        'STATUS',
        'NAME_CONTRACT_STATUS_CAVG'
    ]


    # Label encode categoricals
    cat_cols = input_df.columns[input_df.dtypes == 'object']
    cat_cols = list(set(cat_cols.tolist() + non_obj_categoricals))
    encoder_dict = {}
    small_cat = []
    for c in cat_cols:
        input_df[c] = input_df[c].astype('str')
        if len(input_df[c].unique()) < 5:
            small_cat.append(c)
    print(small_cat)
    # input_df['new_cat'] = input_df['FLAG_OWN_REALTY'] + input_df['CODE_GENDER'] + input_df['NAME_CONTRACT_TYPE'] + input_df['FLAG_OWN_CAR']\
    # + input_df['EMERGENCYSTATE_MODE'] + input_df['HOUSETYPE_MODE'] + input_df['NAME_PRODUCT_TYPE']
    # input_df['new_cat'] = input_df['NAME_EDUCATION_TYPE'] + input_df['OCCUPATION_TYPE']
    # cat_cols.append('new_cat')
    for c in cat_cols:
        if c == 'new_cat':
            print(f'{c} has {len(input_df[c].unique())} unique values')
        encoder = LabelEncoder()
        input_df[c] = encoder.fit_transform(input_df[c].fillna('NULL'))
        encoder_dict[c] = encoder

    return input_df, cat_cols, encoder_dict
def process_dataframe_ori(input_df, nan_as_category = True, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder

    return input_df, categorical_feats.tolist()