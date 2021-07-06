import os
import pandas as pd
import tempfile
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def transform_into_ranking_data(Y_train,Y_test):
    Y_train_not_nan = Y_train.flatten()
    Y_train_not_nan = Y_train_not_nan[~np.isnan(Y_train_not_nan)]
    max_val = np.max(Y_train_not_nan)
    Y_train_not_nan = max_val - Y_train

    Y_test_not_nan = Y_test.flatten()
    Y_test_not_nan = Y_test_not_nan[~np.isnan(Y_test_not_nan)]
    Y_test_not_nan = max_val - Y_test
    
    return Y_train_not_nan, Y_test_not_nan


def sample(Y_train, sample_ratio, seed = 0):
    np.random.seed(seed)
    Y_train_sample = np.copy(Y_train)
    for i in range(Y_train_sample.shape[0]):
        index = np.random.binomial(1, sample_ratio, size=Y_train_sample.shape[1])
        Y_train_sample[i, np.where(index==0)] = np.nan 
    return Y_train_sample


def keepk_sample(Y_train, Y_test, keepk_ratio, keepk, seed = 0,
                 only_keep_training_drugs = False):
    np.random.seed(seed)
    Y_train_keepk = np.copy(Y_train)
    N = int(keepk_ratio * Y_train_keepk.shape[1]) 
    for i in range(Y_train_keepk.shape[0]):
        y = Y_train_keepk[i]
        notnan = ~np.isnan(y)
        y = y[notnan]
        y_argsort = np.argsort(y)[::-1]
        y_argsort_pos = y_argsort[:N]
        y_argsort_neg = y_argsort[N:]
        pos_permutation = np.random.permutation(y_argsort_pos.shape[0])
        for j in range(keepk, pos_permutation.shape[0]):
            y[y_argsort_pos[pos_permutation[j]]] = np.nan
        neg_permutation = np.random.permutation(y_argsort_neg.shape[0])
        for j in range(0, neg_permutation.shape[0]):
            y[y_argsort_neg[neg_permutation[j]]] = np.nan
        Y_train_keepk[i, notnan] = y       

    if only_keep_training_drugs:
        keep = [] 
        for i in range(Y_train_keepk.shape[1]):
            y = Y_train_keepk[:, i]
            if y[~np.isnan(y)].shape[0] > 5:
                keep.append(i)
        keep = np.array(keep)
        

        Y_train_keepk = Y_train_keepk[:, keep]
        Y_test_keepk = Y_test[:, keep]
    else:
        Y_test_keepk = Y_test
    
    return Y_train_keepk, Y_test_keepk



def get_genes_for_network_prop_df(data_frame_path,
    num_genes_per_drug = 10,
    min_weight_gene = None,
    verbose = 0):
    # read data frame
    in_df = pd.read_csv(data_frame_path,sep='\t')
    disable_progressbar = False
    if verbose == 0:
        disable_progressbar = True
        
    
    # collect the prop weights for all the drugs and the genes
    drugs = list(in_df['drug'])
    genes = list(in_df['node'])
    weights = list(in_df['prop_weight'])

    drug_gene_dict = dict()
    for i in tqdm(np.arange(len(drugs)),disable = disable_progressbar):
        cur_drug = drugs[i]
        cur_gene = genes[i]
        cur_weight = weights[i]
        if cur_drug not in drug_gene_dict:
            drug_gene_dict[cur_drug] = dict()
        drug_gene_dict[cur_drug][cur_gene] = cur_weight
    
    
    
    # create list of genes to use by using the top 
    # <num_genes_per_drug> most important genes 
    # per drug higher <min_weight_gene>
    genes_use = []
    for drug in drug_gene_dict.keys():
        cur_drug_dict = drug_gene_dict[drug]
        gene_list = list(cur_drug_dict.keys())
        prop_list = list(cur_drug_dict.values())
        # sort by prop_weight
        sort_ids = np.argsort(prop_list)[::-1]
        if num_genes_per_drug is not None:
            sort_ids = sort_ids[0:num_genes_per_drug]
        for j in range(len(sort_ids)):
            cur_gene = gene_list[sort_ids[j]]
            cur_val = prop_list[sort_ids[j]]
            if min_weight_gene is not None:
                if cur_val >= min_weight_gene:
                    genes_use.append(cur_gene)
            else:
                genes_use.append(cur_gene)
    genes_use = list(set(genes_use))
    return genes_use