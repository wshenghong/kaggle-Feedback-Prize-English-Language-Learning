from cuml.svm import SVR
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
#train emb
label1_train_data = np.load('label1_train_data.npy')
label2_train_data = np.load('label2_train_data.npy')
label3_train_data = np.load('label3_train_data.npy')
label4_train_data = np.load('label4_train_data.npy')
label5_train_data = np.load('label5_train_data.npy')
label6_train_data = np.load('label6_train_data.npy')
#test emb
label1_test_data = np.load('label1_test_data.npy')
label2_test_data = np.load('label2_test_data.npy')
label3_test_data = np.load('label3_test_data.npy')
label4_test_data = np.load('label4_test_data.npy')
label5_test_data = np.load('label5_test_data.npy')
label6_test_data = np.load('label6_test_data.npy')



def choose_train_emb(i, train_):
    if i == 0: tr_text_feats = label1_train_data[list(train_.index), :]
    elif i == 1: tr_text_feats = label2_train_data[list(train_.index), :]
    elif i == 2: tr_text_feats = label3_train_data[list(train_.index), :]
    elif i == 3: tr_text_feats = label4_train_data[list(train_.index), :]
    elif i == 4: tr_text_feats = label5_train_data[list(train_.index), :]
    elif i == 5: tr_text_feats = label6_train_data[list(train_.index), :]
    return tr_text_feats

def choose_test_emb(i):
    if i == 0: te_text_feats = label1_test_data
    elif i == 1: te_text_feats = label2_test_data
    elif i == 2: te_text_feats = label3_test_data
    elif i == 3: te_text_feats = label4_test_data
    elif i == 4: te_text_feats = label5_test_data
    elif i == 5: te_text_feats = label6_test_data
    return te_text_feats

def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)

def main():
    preds = []
    FOLDS = 5
    train = pd.read_csv("train.csv")

    for fold in range(FOLDS):
        train_ = train[train["FOLD"]!=fold]
        test_preds = np.zeros((len(label1_test_data), 6))
        for i,t in enumerate(target_cols):
            tr_text_feats = choose_train_emb(i, train_)
            clf = SVR(C=1)
            clf.fit(tr_text_feats, train_[t].values)
            te_text_feats = choose_test_emb(i)
            test_preds[:,i] = clf.predict(te_text_feats)
        preds.append(test_preds)
    sub = preds.copy()
    sub.loc[:,target_cols] = np.average(np.array(preds),axis=0) #,weights=[1/s for s in scores]
    sub_columns = pd.read_csv("../input/feedback-prize-english-language-learning/sample_submission.csv").columns
    sub = sub[sub_columns]
    sub.to_csv("submission.csv",index=None)