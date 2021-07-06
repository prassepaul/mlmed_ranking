#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import ndcg_score

from ranking_model import Model as Ranker
from ranking_evaluation import RankingEvaluation
from GDSC_ELWC_ranking_tfrecord_writer import create_ELWC_tfrecord 
from GDSC_ELWC_ranking_tfrecord_writer import create_context_dict
from utils import transform_into_ranking_data


def correct_data_from_gt(fold_data, correct_data = True):
    # correct the ground truth
    # while creating the tensorflow record we skip NaN
    # scale the data
    
    columns = np.array(list(fold_data.columns))
    gt_corrected = np.zeros(fold_data.shape)
    use_cols = []
    for i in range(fold_data.shape[0]):
        cur_vals = np.array(fold_data)[i,:]
        if correct_data:
            not_nan_ids = np.where(~np.isnan(cur_vals))[0]
        else:
            not_nan_ids = np.arange(len(cur_vals))
        use_vals = cur_vals[not_nan_ids]
        max_ic50 = np.max(use_vals)
        use_vals = max_ic50 - use_vals
        cur_vals = np.zeros([len(cur_vals),])
        cur_vals[not_nan_ids] = use_vals

        gt_reverse = np.zeros([len(cur_vals),])
        gt_reverse[0:len(use_vals)] = use_vals
        gt_corrected[i,:] = gt_reverse
        use_cols.append(columns[not_nan_ids])
    return gt_corrected, use_cols

def restrict_to_drugs(df,use_drugs):
    dataset_drugs = np.array(list(df.columns))
    use_ids = []
    for i in range(len(use_drugs)):
        try:
            cur_id = int(np.where(use_drugs[i] == dataset_drugs)[0])
        except:
            continue
        use_ids.append(cur_id)

    out_df = df.iloc[:,use_ids]
    return out_df


def convert_to_pathway_outputs(df,pathway_drug_mapping):
    pathways = list(pathway_drug_mapping.keys())
    df_drugs = np.array(df.columns)
    out_vals = np.zeros([df.shape[0],len(pathways)])
    df_vals = df.values
    for i in range(df_vals.shape[0]):
        for j in range(len(pathways)):
            cur_ids = []
            pathway_drugs = list(pathway_drug_mapping[pathways[j]])
            for k in range(len(pathway_drugs)):
                cur_drug = pathway_drugs[k]
                cur_id   = np.where(df_drugs == cur_drug)[0]
                if len(cur_id) == 1:
                    cur_ids.append(int(cur_id))
            mean_val = np.nanmean(df_vals[i,cur_ids])
            out_vals[i,j] = mean_val
    out_df = pd.DataFrame(out_vals)
    out_df.columns = pathways
    out_df.index = df.index
    return out_df



def ndcg_from_file(pred_path,gt_test_path, gt_train_path = None, k=5, exponential_scaling=False, correct_data = True,
                    cell_wise=True, use_drugs = 'all',
                    pathway_drug_mapping = None):       
    fold_test_data = pd.read_csv(gt_test_path,sep=',',index_col=0)
    if pathway_drug_mapping is not None:
        fold_test_data = convert_to_pathway_outputs(fold_test_data,pathway_drug_mapping)
    if gt_train_path is not None:
        fold_train_data = pd.read_csv(gt_train_path,sep=',',index_col=0)
        if pathway_drug_mapping is not None:
            fold_train_data = convert_to_pathway_outputs(fold_train_data,pathway_drug_mapping)
    
        
    if pred_path.endswith('.npy'):
        pred = np.load(pred_path,allow_pickle=True)        
        gt_corrected, use_cols = correct_data_from_gt(fold_test_data,correct_data=correct_data)
    elif pred_path.endswith('.csv'):
        pred = pd.read_csv(pred_path,index_col=0)
        if pathway_drug_mapping is not None:
            pred = convert_to_pathway_outputs(pred,pathway_drug_mapping)
        if use_drugs != 'all':
            pred = restrict_to_drugs(pred,use_drugs)
            fold_train_data = restrict_to_drugs(fold_train_data,use_drugs)
            fold_test_data = restrict_to_drugs(fold_test_data,use_drugs)
        
        pred = np.array(pred.values)
        Y_train = fold_train_data.values
        Y_test = fold_test_data.values

        # transform IC50 into ranking
        _, gt_corrected = transform_into_ranking_data(Y_train,Y_test)
    
    
    if not cell_wise:
        gt_corrected = gt_corrected.transpose()
        pred = pred.transpose()
        
    ndcg  =[]
    for i in range(gt_corrected.shape[0]):
        cur_pred = pred[i,:]
        cur_gt = gt_corrected[i,:]
        cur_gt = np.exp(cur_gt) if exponential_scaling else cur_gt
        not_nan = np.where(~np.isnan(np.array(cur_gt)))[0]
        cur_gt = cur_gt[not_nan]
        cur_pred = cur_pred[not_nan]
        if len(cur_gt) > 1:
            ndcg.append(ndcg_score([cur_gt], [cur_pred], k=k))
    return ndcg


def prec_at_k_from_file(pred_path,gt_test_path, gt_train_path = None, k=5, correct_data = True,
            cell_wise = True, use_drugs = 'all',
                    pathway_drug_mapping = None):
    
    fold_test_data = pd.read_csv(gt_test_path,sep=',',index_col=0)
    if pathway_drug_mapping is not None:
        fold_test_data = convert_to_pathway_outputs(fold_test_data,pathway_drug_mapping)
    if gt_train_path is not None:
        fold_train_data = pd.read_csv(gt_train_path,sep=',',index_col=0)
        if pathway_drug_mapping is not None:
            fold_train_data = convert_to_pathway_outputs(fold_train_data,pathway_drug_mapping)
            
    
    if pred_path.endswith('.npy'):
        pred = np.load(pred_path,allow_pickle=True)
        gt_corrected, use_cols = correct_data_from_gt(fold_test_data,correct_data=correct_data)
    elif pred_path.endswith('.csv'):
        pred = pd.read_csv(pred_path,index_col=0)
        if pathway_drug_mapping is not None:
            pred = convert_to_pathway_outputs(pred,pathway_drug_mapping)
            
        if use_drugs != 'all':
            pred = restrict_to_drugs(pred,use_drugs)
            fold_train_data = restrict_to_drugs(fold_train_data,use_drugs)
            fold_test_data = restrict_to_drugs(fold_test_data,use_drugs)
        
        
        
        pred = np.array(pred.values)        
        Y_train = fold_train_data.values
        Y_test = fold_test_data.values

        # transform IC50 into ranking
        _, gt_corrected = transform_into_ranking_data(Y_train,Y_test)
    
    
    if not cell_wise:
        gt_corrected = gt_corrected.transpose()
        pred = pred.transpose()
    
    """
    The percentage of the true top k elements which
    are predicted as being top k by the model.
    """
    def precision(actual, predicted, k):

        act_set = set(actual[:k])
        pred_set = set(predicted[:k])
        result = len(act_set & pred_set) / float(k)
        return result

    precs  =[]

    for i in range(gt_corrected.shape[0]):
        cur_pred = pred[i,:]
        
        example_rels = np.argsort(-np.array(gt_corrected[i,:]))#[:k]
        example_preds = np.argsort(-np.array(cur_pred))#[:k]
        
        precs.append(precision(example_rels, example_preds, k=k))
        
    return(precs)

def crossvalidate_cv(   cv = 5,
                        cv_split_dir = 'cv_splits/',
                        data_dir = 'data/gdsc_data/',
                        split_nr = 0,
                        filename = None,
                        scoring='paccmann',
                        loss='mse',
                        gene_feature = 'paccmann',
                        model_dir='ranking_model_dir/',                
                        num_train_steps = 1200000,
                        learning_rate = 0.05,
                        save_predictions = True,
                        ks =[1,3,5,10,15,30,50,80],
                        flag_redo = True,
                        cell_wise=True,
                        infix = '_max_conc'):
    
    """
    function to crossvalidate a ranking model given a file containing the train and test data
    
    cv: number of cv folds
    cv_split_dir: directory containing the splits    
    data_dir: directory to gdsc data and location where the features are stored
    split_nr: split to evaluate
    scoring: str, type of scoring function to use "paccmann" or "bl_nn"
    loss: str, "mse" or "approx_ndcg" 
    filename: str, identifier of current run, used to save the files
    num_train_steps: int, number of gradient updates of the training
    learning_rate: float
    ks: list of ints, k values for precision at k/ ndcg at k evaluation
    flag_redo: flag indicating, whether we want to delete the model_dir if it already exists (if True, model_dir will be deleted)
    """
    
    if cell_wise:
        appendix = 'cell_wise'
    else:
        appendix = 'drug_wise'

    filename = str(scoring) + '_' + str(loss) +\
        '_' + str(gene_feature) + '_' + str(cv) + '_' + str(split_nr) + '_' +\
        infix + '_' +\
        appendix        if filename is None else filename

    result = dict()
    if cell_wise:
        train_df_path = cv_split_dir + '/cv_' + str(cv) + '/train_cv_' + str(cv) +\
                '_fold_' + str(split_nr) + infix + '.csv' 
        test_df_path  = cv_split_dir + '/cv_' + str(cv) + '/test_cv_' + str(cv) +\
                '_fold_' + str(split_nr) + infix +  '.csv'
    else:
        train_df_path = cv_split_dir + '/cv_' + str(cv) + '_drug_wise/train_cv_' + str(cv) +\
                '_fold_' + str(split_nr) + infix +  '.csv' 
        test_df_path  = cv_split_dir + '/cv_' + str(cv) + '_drug_wise/test_cv_' + str(cv) +\
                '_fold_' + str(split_nr) + infix +  '.csv'

    train_df = pd.read_csv(train_df_path, index_col=0)
    test_df = pd.read_csv(test_df_path, index_col=0)

    # get train and test contexts
    contexts_train, num_gene_features, num_smiles_features, vocab_size = create_context_dict(train_df,
                        data_dir = data_dir,
                        gene_feature = gene_feature,
                        cell_wise = cell_wise)

    contexts_test, _, _, _ = create_context_dict(test_df,
                        data_dir = data_dir,
                        gene_feature = gene_feature,
                        cell_wise = cell_wise)
    
    if cell_wise:
        n_context_feature = num_gene_features
        n_example_feature = num_smiles_features
        list_size         = train_df.shape[1]
    else:
        n_context_feature = num_smiles_features
        n_example_feature = num_gene_features
        list_size         = train_df.shape[0]


    path_train = "data/tfrecords/"+ filename + "_train.tfrecord"
    path_test  = "data/tfrecords/"+ filename + "_test.tfrecord"

    print("writing train record")
    # create ELWC tfrecords: padding so that each cells list has the same size, needed for tf-ranking
    create_ELWC_tfrecord(contexts_train, filename=path_train,
                         padding=list_size, cell_wise=cell_wise)

    print("writing test record")
    create_ELWC_tfrecord(contexts_test, filename=path_test, 
                         padding=list_size, cell_wise=cell_wise)

    if flag_redo:
        cur_model_dir = model_dir + '/' + str(filename)
        if os.path.isdir(cur_model_dir):
            file_list = os.listdir(cur_model_dir)
            for i in range(len(file_list)):
                try:
                    os.remove(cur_model_dir + '/' + file_list[i])
                except:
                    print('failed to delete ' + cur_model_dir + '/' + file_list[i])

        # run model
        print()
        print()
        print("n_context_features")
        print(n_context_feature)
        print()
        print()
        print()
        print()
        print("n_example_features")
        print(n_example_feature)
        print()
        print()
        ranking_model = Ranker(scoring=scoring,
                                loss=loss,
                                model_dir=cur_model_dir,
                                padding_label=0,
                                label_feature="relevance",
                                n_context_feature=n_context_feature,
                                n_example_feature=n_example_feature,
                                list_size=list_size,
                                cell_wise=cell_wise,
                                smiles_vocabulary_size = vocab_size)  

        ranking_model.train(learning_rate=learning_rate, 
                              num_train_steps=num_train_steps,
                              train_data_path=path_train,
                              eval_data_path=None)


        # predictions
        predictions = ranking_model.predict(test_size = len(contexts_test),
                                              test_data_path = path_test)

        ### save as *.npy with row and column
        # TODO:
        if(save_predictions):
            np.save('data/preds/pred_test_' + str(filename) + '.npy', predictions)

        # evaluation
        rank_eval = RankingEvaluation(predictions=predictions, test_tfrecord_path=path_test)

        
        # save csv
        if cell_wise:
            out_df_values = np.ones(test_df.shape) * np.nan

            index_id_dict = dict()
            index_list = list(test_df.index)
            for ii in range(len(index_list)):
                index_id_dict[index_list[ii]] = ii

            columns_id_dict = dict()
            columns_list = list(test_df.columns)
            for ii in range(len(columns_list)):
                columns_id_dict[columns_list[ii]] = ii

            context_test_keys = list(contexts_test.keys())
            for i in range(len(context_test_keys)):
                cur_key = context_test_keys[i]
                cur_data = contexts_test[cur_key]['examples']
                cell_names = list(cur_data['drug_name'])
                cur_i = index_id_dict[cur_key]
                for j in range(len(cell_names)):
                    cur_j = columns_id_dict[cell_names[j]]        
                    out_df_values[cur_i,cur_j] = predictions[i][j]

            out_df = pd.DataFrame(out_df_values)
            out_df.columns = test_df.columns
            out_df.index = test_df.index
            out_df.to_csv('data/preds/pred_test_' + str(filename) + '.csv',sep=',')
        else:
            out_df_values = np.ones(test_df.shape) * np.nan

            index_id_dict = dict()
            index_list = list(test_df.index)
            for ii in range(len(index_list)):
                index_id_dict[index_list[ii]] = ii

            columns_id_dict = dict()
            columns_list = list(test_df.columns)
            for ii in range(len(columns_list)):
                columns_id_dict[columns_list[ii]] = ii

            context_test_keys = list(contexts_test.keys())
            for i in range(len(context_test_keys)):
                cur_key = context_test_keys[i]
                cur_data = contexts_test[cur_key]['examples']
                cell_names = list(cur_data['cell_name'])
                cur_i = columns_id_dict[cur_key]
                for j in range(len(cell_names)):
                    cur_j = index_id_dict[cell_names[j]]        
                    out_df_values[cur_j,cur_i] = predictions[i][j]

            out_df = pd.DataFrame(out_df_values)
            out_df.columns = test_df.columns
            out_df.index = test_df.index
            out_df.to_csv('data/preds/pred_test_' + str(filename) + '.csv',sep=',')

        
        ndcg_exp_dict = {}
        ndcg_lin_dict = {}
        prec_at_k_dict = {}

        for k in ks:
            ndcg_exp_dict[k]  = {}
            ndcg_lin_dict[k]  = {}
            prec_at_k_dict[k] = {}

            ndcgs = rank_eval.ndcg(k=k, exponential_scaling=True)
            ndcg_exp_dict[k]["mean"] = np.mean(ndcgs)
            ndcg_exp_dict[k]["std"]  = np.std(ndcgs)

            ndcgs = rank_eval.ndcg(k=k, exponential_scaling=False)
            ndcg_lin_dict[k]["mean"] = np.mean(ndcgs)
            ndcg_lin_dict[k]["std"]  = np.std(ndcgs)

            precs_at_k = rank_eval.prec_at_k(k=k)
            prec_at_k_dict[k]["mean"] = np.mean(precs_at_k)
            prec_at_k_dict[k]["std"]  = np.std(precs_at_k)


        result["ndcg_exp"]  = ndcg_exp_dict
        result["ndcg_lin"]  = ndcg_lin_dict
        result["prec_at_k"] = prec_at_k_dict
        
    
    # try to remove tensorflow records
    try:
        os.remove(path_train)
        os.remove(path_test)
    except:
        pass
        
    return result



def crossvalidate(context_dict,
                n_splits = 5,
                filename = None,
                scoring="paccmann",
                loss="mse",
                model_dir="ranking_model_dir",
                n_context_feature=2128,
                n_example_feature=155,
                list_size=390,
                num_train_steps = 1200000,
                learning_rate = 0.05,
                save_predictions = True,
                ks =[1,3,5,10,15,30,50,80,390],
                only_execute_fold_nr = None,
                flag_redo = True,
                cell_wise=True
                 ):
    
    """
    function to crossvalidate a ranking model
    
    contexts: list of ints, cell keys of context dict
    n_splits: number of cv folds
    scoring: str, type of scoring function to use "paccmann" or "bl_nn"
    loss: str, "mse" or "approx_ndcg" 
    filename: str, identifier of current run, used to save the files
    n_context_feature: int, number of cell features
    n_example_feature: int, number of drug features
    list_size: int, size of the largest example(drug list) of the cells, all example lists are padded to this size
    num_train_steps: int, number of gradient updates of the training
    learning_rate: float
    ks: list of ints, k values for precision at k/ ndcg at k evaluation
    only_execute_fold_nr: index (starting at 1) of fold to execute (all other folds are skipped)
    flag_redo: flag indicating, whether we want to delete the model_dir if it already exists (if True, model_dir will be deleted)
    """
    contexts = np.array(list(context_dict.keys()))

    filename = scoring + loss if filename is None else filename
    
    

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    cv_generator = kf.split(contexts)
    results = {}
    
    fold = 1
    for train_ind, test_ind in cv_generator:
        # skip fold if only_execute_fold_nr is not 
        # None and only_execute_fold_nr != fold_nr
        if only_execute_fold_nr is not None:
            if fold != only_execute_fold_nr:
                fold += 1
                print('skip fold nr: ' + str(fold-1))
                continue
             
        results["fold"+str(fold)] = {}
        #print(f"processing fold: {fold}/{n_splits}")
        contexts_train = contexts[train_ind]  
        contexts_test  = contexts[test_ind]  

        # create sub dictionaries with train/test contexts
        context_dict_train ={k:context_dict[k] for k in contexts_train}
        context_dict_test = {k:context_dict[k] for k in contexts_test}

        np.save('data/tfrecord_annot/' + filename + '_fold' + str(fold) + '_dict_train.npy', context_dict_train)
        np.save('data/tfrecord_annot/' + filename + '_fold' + str(fold) + '_dict_test.npy', context_dict_test)

        path_train = 'data/tfrecords/' + filename + '_fold' + str(fold) + '_train.tfrecord'
        path_test  = 'data/tfrecords/' + filename + '_fold' + str(fold) + '_test.tfrecord'

        print("writing train record")
        # create ELWC tfrecords: padding so that each cells list has the same size, needed for tf-ranking
        create_ELWC_tfrecord(context_dict_train, filename=path_train,
                             padding=list_size, cell_wise=cell_wise)

        print("writing test record")
        create_ELWC_tfrecord(context_dict_test, filename=path_test, 
                             padding=list_size, cell_wise=cell_wise)

        if flag_redo:
            cur_model_dir = model_dir+str(fold)
            if os.path.isdir(cur_model_dir):
                file_list = os.listdir(cur_model_dir)
                for i in range(len(file_list)):
                    try:
                        os.remove(cur_model_dir + '/' + file_list[i])
                    except:
                        print('failed to delete ' + cur_model_dir + '/' + file_list[i])
                    
        # run model
        print()
        print()
        print("n_context_features")
        print(n_context_feature)
        print()
        print()
        print()
        print()
        print("n_example_features")
        print(n_example_feature)
        print()
        print()
        ranking_model = Ranker(scoring=scoring,
                                loss=loss,
                                model_dir=cur_model_dir,
                                padding_label=0,
                                label_feature="relevance",
                                n_context_feature=n_context_feature,
                                n_example_feature=n_example_feature,
                                list_size=list_size,
                                cell_wise=cell_wise)  

        ranking_model.train(learning_rate=learning_rate, 
                              num_train_steps=num_train_steps,
                              train_data_path=path_train,
                              eval_data_path=None)


        # predictions
        predictions = ranking_model.predict(test_size = len(context_dict_test),
                                              test_data_path = path_test)
        if(save_predictions):
            np.save('data/preds/pred_test' + str(model_dir) + '.npy', predictions)    

        # evaluation
        rank_eval = RankingEvaluation(predictions=predictions, test_tfrecord_path=path_test)


        ndcg_exp_dict = {}
        ndcg_lin_dict = {}
        prec_at_k_dict = {}

        for k in ks:
            ndcg_exp_dict[k]  = {}
            ndcg_lin_dict[k]  = {}
            prec_at_k_dict[k] = {}

            ndcgs = rank_eval.ndcg(k=k, exponential_scaling=True)
            ndcg_exp_dict[k]["mean"] = np.mean(ndcgs)
            ndcg_exp_dict[k]["std"]  = np.std(ndcgs)

            ndcgs = rank_eval.ndcg(k=k, exponential_scaling=False)
            ndcg_lin_dict[k]["mean"] = np.mean(ndcgs)
            ndcg_lin_dict[k]["std"]  = np.std(ndcgs)

            precs_at_k = rank_eval.prec_at_k(k=k)
            prec_at_k_dict[k]["mean"] = np.mean(precs_at_k)
            prec_at_k_dict[k]["std"]  = np.std(precs_at_k)


        results["fold"+str(fold)]["ndcg_exp"]  = ndcg_exp_dict
        results["fold"+str(fold)]["ndcg_lin"]  = ndcg_lin_dict
        results["fold"+str(fold)]["prec_at_k"] = prec_at_k_dict
        
        fold += 1
    return results

