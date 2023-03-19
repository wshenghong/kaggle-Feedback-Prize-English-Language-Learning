
import numpy as np 
import pandas as pd 
import os, gc, re
import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import generate_emb

FOLDS = 10
BATCH_SIZE = 4
tokenizer = None
MAX_LEN = 640

LM_path = {
    'deberta-base': 'deberta-base/deberta-base',
    'deberta-v3-large': 'deberta-v3-large/deberta-v3-large',
    'deberta-large': 'deberta-large/deberta-large',
    'deberta-large-mnli': 'deberta-large-mnli/deberta-large-mnli',
    'deberta-xlarge': 'deberta-xlarge/deberta-xlarge',
    'deberta-v3-base': 'debertav3base/deberta-v3-base',
    'deberta-v3-small': 'deberta-v3-small/deberat-v3-small'
}

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens


def main():
    #read data
    train = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/train.csv")
    train["src"]="train"
    test = pd.read_csv("/kaggle/input/feedback-prize-english-language-learning/test.csv")
    test["src"]="test"
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    kfold = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for i,(train_index, val_index) in enumerate(kfold.split(train, train[target_cols])):
        train.loc[val_index,'FOLD'] = i
    #emb_dataset
    training_set = EmbedDataset(train)
    testing_set = EmbedDataset(test)
    #emb_dataloaber
    train_emb_dataloader = torch.utils.data.DataLoader(training_set,\
                            batch_size=BATCH_SIZE,\
                            shuffle=False) 
    test_embed_dataloader = torch.utils.data.DataLoader(testing_set,\
                            batch_size=BATCH_SIZE,\
                            shuffle=False)                    

    #training_embeddings&testing_embeddings
    #變數命名1_2代表用於label1,2，v3_base為版本，最後面為embedding大小
    #label1、2、3、4、5、6
    train_label1_2_v3_base_500, test_label1_2_v3_base_500 = generate_emb.generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-v3-base'], MAX=500)
    train_label1_2_3_6_v3_large_600, test_label1_2_3_6_v3_large_600 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-v3-large'], MAX=600)
    train_label1_2_3_4_5_6_large_500, test_label1_2_3_4_5_6_large_500 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-large'], MAX=500)
    train_label1_2_v3_small_500, test_label1_2_v3_small_500 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberat-v3-small'], MAX=500)
    train_label1_2_3_4_5_mnli_500, test_label1_2_3_4_5_mnli_500 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-large-mnli'], MAX=500)
    #label3、4
    train_label3_4_v3_base_800, test_label3_4_v3_base_800 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-v3-base'], MAX=800)
    train_label3_v3_small_800, test_label3_v3_small_800 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberat-v3-small'], MAX=800)
    #label4、5、6
    train_label4_5_6_v4_base_600, test_label4_5_6_v4_base_600 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberat-v3-small'], MAX=600)
    train_label4_5_v3_large_700, test_label4_5_v3_large_700 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-v3-large'], MAX=700)
    train_label4_5_6_v4_small_600, test_label4_5_6_v4_small_600 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberat-v3-small'], MAX=600)
    train_label4_6_mnli_600, test_label4_6_mnli_600 = generate_emb.get_embeddings(train_emb_dataloader, test_embed_dataloader, LM_path['deberta-large-mnli'], MAX=600)

    #concatenate for every train_set label
    label1_train_data = np.concatenate([train_label1_2_v3_base_500, train_label1_2_3_6_v3_large_600,
                                    train_label1_2_3_4_5_6_large_500, train_label1_2_v3_small_500, 
                                    train_label1_2_3_4_5_mnli_500], axis=1)
    label2_train_data = np.concatenate([train_label1_2_v3_base_500, train_label1_2_3_6_v3_large_600,
                                    train_label1_2_3_4_5_6_large_500, train_label1_2_v3_small_500, 
                                    train_label1_2_3_4_5_mnli_500], axis=1)
    label3_train_data = np.concatenate([train_label3_4_v3_base_800, train_label1_2_3_6_v3_large_600,
                                    train_label1_2_3_4_5_6_large_500, train_label3_v3_small_800, 
                                    train_label1_2_3_4_5_mnli_500], axis=1)
    label4_train_data = np.concatenate([train_label4_5_6_v4_base_600, train_label4_5_v3_large_700,
                                    train_label1_2_3_4_5_6_large_500, train_label4_5_6_v4_small_600, 
                                    train_label1_2_3_4_5_mnli_500], axis=1)
    label5_train_data = np.concatenate([train_label4_5_6_v4_base_600, train_label4_5_v3_large_700,
                                    train_label1_2_3_4_5_6_large_500, train_label4_5_6_v4_small_600, 
                                    train_label1_2_3_4_5_mnli_500], axis=1)
    label6_train_data = np.concatenate([train_label4_5_6_v4_base_600, train_label1_2_3_6_v3_large_600,
                                    train_label1_2_3_4_5_6_large_500, train_label4_5_6_v4_small_600, 
                                    train_label4_6_mnli_600], axis=1)

    #concatenate for every test_set label
    label1_test_data = np.concatenate([test_label1_2_v3_base_500, test_label1_2_3_6_v3_large_600,
                                    test_label1_2_3_4_5_6_large_500, test_label1_2_v3_small_500, 
                                    test_label1_2_3_4_5_mnli_500], axis=1)
    label2_test_data = np.concatenate([test_label1_2_v3_base_500, test_label1_2_3_6_v3_large_600,
                                    test_label1_2_3_4_5_6_large_500, test_label1_2_v3_small_500, 
                                    test_label1_2_3_4_5_mnli_500], axis=1)
    label3_test_data = np.concatenate([test_label3_4_v3_base_800, test_label1_2_3_6_v3_large_600,
                                    test_label1_2_3_4_5_6_large_500, test_label3_v3_small_800, 
                                    test_label1_2_3_4_5_mnli_500], axis=1)
    label4_test_data = np.concatenate([test_label4_5_6_v4_base_600, test_label4_5_v3_large_700,
                                    test_label1_2_3_4_5_6_large_500, test_label4_5_6_v4_small_600, 
                                    test_label1_2_3_4_5_mnli_500], axis=1)
    label5_test_data = np.concatenate([test_label4_5_6_v4_base_600, test_label4_5_v3_large_700,
                                    test_label1_2_3_4_5_6_large_500, test_label4_5_6_v4_small_600, 
                                    test_label1_2_3_4_5_mnli_500], axis=1)
    label6_test_data = np.concatenate([test_label4_5_6_v4_base_600, test_label1_2_3_6_v3_large_600,
                                    test_label1_2_3_4_5_6_large_500, test_label4_5_6_v4_small_600, 
                                    test_label4_6_mnli_600], axis=1)

    #save embeddings
    #train emb
    np.save('label1_train_data.npy', label1_train_data)
    np.save('label2_train_data.npy', label2_train_data)
    np.save('label3_train_data.npy', label3_train_data)
    np.save('label4_train_data.npy', label4_train_data)
    np.save('label5_train_data.npy', label5_train_data)
    np.save('label6_train_data.npy', label6_train_data)
    #test emb
    np.save('label1_test_data.npy', label1_test_data)
    np.save('label2_test_data.npy', label2_test_data)
    np.save('label3_test_data.npy', label3_test_data)
    np.save('label4_test_data.npy', label4_test_data)
    np.save('label5_test_data.npy', label5_test_data)
    np.save('label6_test_data.npy', label6_test_data)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
