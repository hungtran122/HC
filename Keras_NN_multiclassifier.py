import os
import pandas as pd
import numpy as np
from libs.utils import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.utils import plot_model

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

input_dir = os.path.join(os.getcwd(), 'input')
n_folds = 5
model_name = 'dnn'
n_classes = 2


print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
sample_size = None
if sample_size is None:
    checkpoint_path = './checkpoint'
else:
    checkpoint_path = './checkpoint/rubbish'

if sample_size is None:
    submission_dir = './submission'
else:
    submission_dir = './submission/rubbish'
# Fixed graph parameters
# EMBEDDING_SIZE = 3  # Use cardinality / 2 instead
N_HIDDEN_1 = 160
N_HIDDEN_2 = 128
N_HIDDEN_3 = 64
N_HIDDEN_4 = 64
# Learning parameters
LR = 0.01
N_EPOCHS = 30
BATCH_SIZE = 256

def nn_model(X_num, X_cat, cat_len, cat_cols):
    input_cat = []
    out_cat = []
    print(X_cat.shape[1])
    for name in cat_cols:
        input_cat.append(Input(shape=(1,), name="cat_" + name))
    for x, c_len in zip(input_cat, cat_len):
        if c_len < 8:
            embed_size = 5
        else:
            embed_size = (c_len//2) + 1
        if embed_size > 100:
            embed_size = 100
        x = Embedding(c_len + 1, embed_size, embeddings_initializer='he_normal')(x)
        # x = SpatialDropout1D(0.25)(x)
        x = Flatten()(x)
        x = Dense(c_len, activation=None, kernel_initializer='he_normal')(x)
        x = PReLU()(x)
        out_cat.append(x)
    out_cat = concatenate(out_cat, axis=-1)
    out_cat = Dense(32, activation="relu", kernel_initializer='he_normal')(out_cat)

    input_num = Input(shape=(X_num.shape[1],), name="numeric")
    out_num = Dense(32, activation="relu", kernel_initializer='he_normal')(input_num)
    out_num = BatchNormalization()(out_num)
    out_num = Dropout(0.5)(out_num)

    out_num_cat = concatenate([out_cat, input_num], axis=-1)
    print(out_num_cat.shape)
    out_num_cat = Dense(N_HIDDEN_1, activation=None, kernel_initializer='he_normal')(out_num_cat)
    out_num_cat = PReLU()(out_num_cat)
    out_num_cat = BatchNormalization()(out_num_cat)
    out_num_cat = Dropout(0.5)(out_num_cat)

    out_num_cat = Dense(N_HIDDEN_2, activation=None, kernel_initializer='he_normal')(out_num_cat)
    out_num_cat = PReLU()(out_num_cat)
    out_num_cat = BatchNormalization()(out_num_cat)
    out_num_cat = Dropout(0.5)(out_num_cat)

    out_num_cat = Dense(N_HIDDEN_3, activation=None, kernel_initializer='he_normal')(out_num_cat)
    out_num_cat = PReLU()(out_num_cat)
    out_num_cat = BatchNormalization()(out_num_cat)
    out_num_cat = Dropout(0.5)(out_num_cat)

    # out_num_cat = Dense(N_HIDDEN_4, activation=None, kernel_initializer='he_normal')(out_num_cat)
    # out_num_cat = PReLU()(out_num_cat)
    # out_num_cat = BatchNormalization()(out_num_cat)
    # out_num_cat = Dropout(0.5)(out_num_cat)

    print(out_num_cat.shape)
    # out_num_cat = Dense(1, activation="sigmoid", kernel_initializer='he_normal')(out_num_cat)
    out_num_cat = Dense(n_classes, activation="softmax", kernel_initializer='he_normal')(out_num_cat)
    model = Model(inputs=[*input_cat, input_num], outputs=out_num_cat)
    return model

def train():
    app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv.zip'), nrows=sample_size)
    if sample_size is None:
        len_train = len(app_train_df)
        merged_df = pd.read_csv(os.path.join(input_dir, 'data_feature_engineering.csv'), nrows=sample_size)
    else:
        len_train = sample_size
        merged_df = pd.read_csv(os.path.join(input_dir, 'data_feature_engineering.csv'), nrows=sample_size * 2)


    # Separate metadata
    meta_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']
    meta_df = merged_df[meta_cols]
    merged_df.drop(meta_cols, axis=1, inplace=True)

    # Process the data set.
    with timer('Processing data frame'):
        merged_df, cat_cols, encoder_dict = process_dataframe(input_df=merged_df)

    # Capture other categorical features not as object data types:
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
        # 'NUM_INSTALMENT_VERSION',
        # 'NUM_INSTALMENT_NUMBER',
        'NAME_CONTRACT_STATUS_CAVG'
    ]
    # print('Categorical features length before add non object categorical features: ', len(cat_cols))
    # cat_cols = list(set(cat_cols + non_obj_categoricals))
    # print('Categorical features length after add non object categorical features: ', len(cat_cols))
    for c in cat_cols:
        print(f'{c} has {len(merged_df[c].unique())} unique values')

    y = merged_df[:len_train].TARGET
    le = preprocessing.LabelEncoder()
    allvalues = y.unique().tolist()
    le.fit(allvalues)
    y = le.transform(y.values)
    # y = to_categorical(y)
    merged_df.drop('TARGET', axis=1, inplace=True)

    null_counts = merged_df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    null_ratios = null_counts / len(merged_df)

    # Drop columns over x% null
    null_thresh = .8
    null_cols = null_ratios[null_ratios > null_thresh].index
    merged_df.drop(null_cols, axis=1, inplace=True)
    print('Columns dropped for being over {}% null:'.format(100 * null_thresh))
    for col in null_cols:
        print(col)
        if col in cat_cols:
            cat_cols.pop(col)

    # Fill the rest with the mean (TODO: do something better!)
    # merged_df.fillna(merged_df.median(), inplace=True)
    merged_df.fillna(0, inplace=True)


    cat_feats_idx = np.array([merged_df.columns.get_loc(x) for x in cat_cols])
    num_cols = list(set(merged_df.columns.tolist()) - set(cat_cols + ['index']))
    int_feats_idx = [merged_df.columns.get_loc(x) for x in non_obj_categoricals]
    cat_feat_lookup = pd.DataFrame({'feature': cat_cols, 'column_index': cat_feats_idx})
    cat_feat_lookup.head()

    cont_feats_idx = np.array(
        [merged_df.columns.get_loc(x)
         for x in merged_df.columns[~merged_df.columns.isin(cat_cols)]]
    )
    cont_feat_lookup = pd.DataFrame(
        {'feature': merged_df.columns[~merged_df.columns.isin(cat_cols)],
         'column_index': cont_feats_idx}
    )
    merged_df.replace({np.inf:999, -np.inf:-999}, inplace=True)

    scaler = StandardScaler()
    final_col_names = merged_df.columns
    merged_df[num_cols] = scaler.fit_transform(merged_df[num_cols])

    # scaler_2 = MinMaxScaler(feature_range=(0, 1))
    # merged_df[non_obj_categoricals] = scaler_2.fit_transform(merged_df[non_obj_categoricals])

    # Re-separate into labelled and unlabelled
    tr_df = merged_df[:len_train]
    te_df = merged_df[len_train:]
    print('Shape of test data frame', te_df.shape)
    X_num = np.array(tr_df[num_cols])
    X_cat = np.array(tr_df[cat_cols])
    X_te_num = np.array(te_df[num_cols])
    X_te_cat = [te_df[c] for c in cat_cols]
    np.save(os.path.join(input_dir,'X_te_num'), X_te_num)
    np.save(os.path.join(input_dir, 'X_te_cat'), X_te_cat)

    n_cont_inputs = merged_df[num_cols].shape[1]


    print('Number of continous features: ', n_cont_inputs)
    print('Number of categoricals pre-embedding: ', merged_df[cat_cols].shape[1])

    cat_len = []
    for c in cat_cols:
        try:
            cat_len.append(len(merged_df[c].unique()))
        except KeyError:
            cat_cols.remove(c)



    early = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=1e-4)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=3,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')

    def roc_curve(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    model = nn_model(X_num, X_cat, cat_len, cat_cols)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=LR))  # , metrics=['binary_accuracy'])
    model.summary()

    # kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
    skf = StratifiedKFold(n_splits=n_folds, random_state=2)
    skf.get_n_splits(tr_df, y)
    oof_preds = np.zeros((tr_df.shape[0], 1))
    le = preprocessing.LabelEncoder()
    le.fit(y.tolist())

    for fold, (train_index, val_index) in enumerate(skf.split(tr_df, y)):
        y_onehot = to_categorical(y)
        print(f"\n[+] Fold {fold}")
        file_path = f"{checkpoint_path}/keras_{model_name}_weights_{fold}.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        csv_logger = CSVLogger(f'{checkpoint_path}/log_{model_name}_{fold}.csv', append=True, separator=',')
        callbacks_list = [checkpoint, early,  csv_logger]

        X_tr_num = np.array(tr_df[num_cols])[train_index]
        X_tr_cat = [tr_df[c][train_index] for c in cat_cols]
        y_tr = y_onehot[train_index]

        print(X_tr_num.shape)
        print(len(X_tr_cat))

        X_val_num = np.array(tr_df[num_cols])[val_index]
        X_val_cat = [tr_df[c][val_index] for c in cat_cols]
        y_val = y_onehot[val_index]

        history = model.fit([*X_tr_cat, X_tr_num], y_tr,
                            validation_data=([*X_val_cat, X_val_num], y_val),
                            verbose=1, callbacks=callbacks_list,
                            epochs=N_EPOCHS, batch_size=BATCH_SIZE)
        model.load_weights(file_path)
        model_path = f"{checkpoint_path}/keras_{model_name}_{fold}.h5"
        model.save(model_path)
        preds = model.predict([*X_val_cat, X_val_num], batch_size=BATCH_SIZE)
        oof_preds[val_index] = preds[:,1].reshape(y_val.shape[0],1)


        # preds = np.argmax(oof_preds[val_index], axis=1)
        # preds = le.inverse_transform(preds)
        # print('*****************')
        # print(type(y_val), y_val.shape)

        print('Fold %2d AUC : %.6f' % (fold + 1, roc_auc_score(y[val_index], oof_preds[val_index])))
    print('OOF AUC : %.6f' % (roc_auc_score(y, oof_preds)))
def submit():
    preds_all = []
    for fold in range(n_folds):
        print(f"\n[+++++] Making prediction on fold {fold+1}")
        model_path = f"{checkpoint_path}/keras_{model_name}_{fold}.h5"
        weight_path = f"{checkpoint_path}/keras_{model_name}_weights_{fold}.h5"
        model = keras.models.load_model(model_path)
        # model.load_weights(weight_path)
        X_te_num = np.load(os.path.join(input_dir, 'X_te_num.npy'))
        X_te_cat = np.load(os.path.join(input_dir, 'X_te_cat.npy'))

        pred_ = model.predict([*X_te_cat, X_te_num], batch_size=BATCH_SIZE)
        pred = pred_[:, 1].reshape(pred_.shape[0], 1)
        submission = pd.read_csv("./submission/sample_submission.csv", nrows = sample_size)
        submission['TARGET'] = pred
        submission.to_csv(os.path.join(submission_dir, f'keras_fold_{fold + 1}_submission.csv'), index=False)
        preds_all.append(pred)
    preds_all = np.array(preds_all)
    preds_avg = np.mean(preds_all, axis=0)
    out_df = pd.read_csv("./submission/sample_submission.csv", nrows = sample_size)
    out_df['TARGET'] = preds_avg
    out_df.to_csv(os.path.join(submission_dir, f'keras_avg_{n_folds}_folds_submission.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'submit'])
    args = parser.parse_args()
    print(f"[+++++] Start {args.mode} mode")
    if args.mode == 'train':
        train()
    elif args.mode == 'submit':
        submit()







