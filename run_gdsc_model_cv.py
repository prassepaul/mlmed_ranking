from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
#from tensorflow_serving.apis import input_pb2
import os
import ranking_cv as r_cv
from ranking_model import Model as Ranker
import six
import os
import numpy as np
import sys
import argparse
import tensorflow_ranking as tfr

def boolean_string(s):
    if s == 'True\r':
        s = 'True'
    elif s == 'False\r':
        s='False'
    
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', type=int, default=0)    
    parser.add_argument('-cv_split_dir', '--cv_split_dir', type=str, default='cv_splits/')
    parser.add_argument('-data_dir', '--data_dir', type=str, default='data/gdsc_data/')
    parser.add_argument('-filename', '--filename', type=str, default=None)
    parser.add_argument('-scoring', '--scoring', type=str, default='paccmann', help = 'Set type of model possible values are \'paccmann\' and \'nn_baseline\'')
    parser.add_argument('-loss', '--loss', type=str, default='approx_ndcg', help = 'Set type of loss function; possible values are \'approx_ndcg\' and \'mse\'')
    parser.add_argument('-gene_feature', '--gene_feature', type=str, default='paccmann', help = 'Set type of features used to train the model; possible values are \'paccmann\', \'all_gene\', \'netcore_sig_gdsc_drug_targets_genes_10\', \'netcore_sig_gdsc_drug_targets_genes_20\', \'netcore_sig_gdsc_drug_targets_genes_30\', \'netcore_sig_gdsc_drug_targets_literature_mining_genes_10\', \'netcore_sig_gdsc_drug_targets_literature_mining_genes_20\', \'netcore_sig_gdsc_drug_targets_literature_mining_genes_30\', \'netcore_sig_literature_mining_genes_10\', \'netcore_sig_literature_mining_genes_20\',and \'netcore_sig_literature_mining_genes_30\'')
    parser.add_argument('-model_dir', '--model_dir', type=str, default='ranking_model_dir/')
    parser.add_argument('-num_train_steps', '--num_train_steps', type=int, default=1000000)
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=0.05)
    parser.add_argument('-flag_redo', '--flag_redo', type=boolean_string, default=False)
    parser.add_argument('-cell_wise', '--cell_wise', type=boolean_string, default=True)
    parser.add_argument('-infix', '--infix', type=str, default='', help='Set type of ground truth used possible values are \'\' for (ic50) and \'max_conc\' for using the normalized ic50 (ic50 / max_conc=')
    parser.add_argument('-num_folds', '--num_folds', type=int, default=5)
    parser.add_argument('-model_suffix', '--model_suffix', type=str, default='')
    parser.add_argument('-use_drugs', '--use_drugs', type=str, default='all', help='Set the list of drugs used to train, default is \'all\', e.g. [\'Paclitaxel\',\'Erlotinib\',\'Cetuximab\']')
    parser.add_argument('-pred_dir','--pred_dir', type=str, default='results/preds/')
    parser.add_argument('-fold_nr','--fold_nr', type=int, default = -1)
    parser.add_argument('-drug_type','--drug_type', type=str, default='all')
    parser.add_argument('-drug_type_path','--drug_type_path',type=str, default='data/gdsc_data/drugs.txt')
    parser.add_argument('-drug_repurposing', '--drug_repurposing', type=boolean_string, default=False)
    parser.add_argument('-tf_record_dir','--tf_record_dir',type=str, default = 'data/tfrecords/')
    
    args = parser.parse_args()    
    
    GPU = args.gpu
    cv_split_dir = args.cv_split_dir
    data_dir = args.data_dir
    filename = args.filename
    scoring = args.scoring
    loss = args.loss
    gene_feature = args.gene_feature
    model_dir = args.model_dir
    num_train_steps = args.num_train_steps
    learning_rate = args.learning_rate
    flag_redo = args.flag_redo
    cell_wise = args.cell_wise
    infix = args.infix
    num_folds = args.num_folds
    use_drugs = args.use_drugs    
    model_suffix = args.model_suffix
    pred_dir = args.pred_dir
    fold_nr = args.fold_nr
    drug_type = args.drug_type
    drug_type_path = args.drug_type_path
    drug_repurposing = args.drug_repurposing
    tf_record_dir = args.tf_record_dir
    
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)
    
    save_predictions = True
    
    import socket
    print(socket.gethostname())
    
    
    # select graphic card    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)
    
    
    if cell_wise:
        appendix = 'cell_wise'
    else:
        appendix = 'drug_wise'
        if drug_repurposing:
            appendix += '_drug_repurposing'
    
    appendix += model_suffix
    
    filename = str(scoring) + '_' + str(loss) +\
        '_' + str(gene_feature) + '_' +\
        infix + '_' +\
        appendix        if filename is None else filename
        
    
    if len(infix) > 0:
        infix = '_' + infix
        
    # collect test data
    for split_nr in range(num_folds):
        if fold_nr != -1:
            split_nr = fold_nr
        
        if cell_wise or drug_repurposing:
            train_df_path = cv_split_dir + '/cv_' + str(num_folds) + '/train_cv_' + str(num_folds) +\
                    '_fold_' + str(split_nr) + infix + '.csv' 
            test_df_path  = cv_split_dir + '/cv_' + str(num_folds) + '/test_cv_' + str(num_folds) +\
                    '_fold_' + str(split_nr) + infix +  '.csv'
        else:
            train_df_path = cv_split_dir + '/cv_' + str(num_folds) + '_drug_wise/train_cv_' + str(num_folds) +\
                    '_fold_' + str(split_nr) + infix +  '.csv' 
            test_df_path  = cv_split_dir + '/cv_' + str(num_folds) + '_drug_wise/test_cv_' + str(num_folds) +\
                    '_fold_' + str(split_nr) + infix +  '.csv'
        
        
        csv_save_path = pred_dir + '/pred_test_' + filename + '_' + str(split_nr) + '.csv'
        np_save_path = pred_dir + 'pred_test_' +  filename + '_' + str(split_nr) + '.npy'
        
        if os.path.exists(csv_save_path) and not flag_redo:
            continue
                
        train_df = pd.read_csv(train_df_path, index_col=0)
        test_df = pd.read_csv(test_df_path, index_col=0)
                
        if use_drugs != 'all' or drug_type != 'all':
            if use_drugs != 'all':
                drug_list = use_drugs.replace('[','').replace(']','').replace('\'','').split(',')
            else:
                drug_type_data = pd.read_csv(drug_type_path,sep='\t',header=None)
                drug_list = list(drug_type_data[0][drug_type_data[1] == drug_type])
        
            # select drugs we want to use the train the model    
            column_names = np.array(list(test_df.columns))

            use_col_ids = []
            used_drugs = []
            for d_i in range(len(drug_list)):
                cur_drug = np.str(drug_list[d_i].strip())
                cur_id = np.where(column_names == cur_drug)

                if len(cur_id) > 0:
                    try:
                        cur_id = cur_id[0]
                        use_col_ids.append(int(cur_id))
                        used_drugs.append(cur_drug)
                    except:
                        pass
            if len(used_drugs) < 2:
                print('the number of drugs that was found is: ' + str(len(used_drugs)) + ' ... ABORT')
                return -1
            test_df = test_df.iloc[:,use_col_ids]
            train_df = train_df.iloc[:,use_col_ids]
            
        # get train and test contexts
        contexts_train, num_gene_features, num_smiles_features, vocab_size = r_cv.create_context_dict(train_df,
                            data_dir = data_dir,
                            gene_feature = gene_feature,
                            cell_wise = cell_wise)

        contexts_test, _, _, _ = r_cv.create_context_dict(test_df,
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
        
        
        if cell_wise:
            n_context_feature = num_gene_features
            n_example_feature = num_smiles_features
            list_size         = test_df.shape[1]
        else:
            n_context_feature = num_smiles_features
            n_example_feature = num_gene_features
            list_size         = test_df.shape[0]
        
        path_train = tf_record_dir + filename + '_train_' + str(split_nr) + '.tfrecord'
        path_test  = tf_record_dir + filename + '_test' + str(split_nr) + '.tfrecord'
        
        
        
        print("writing train record")
        # create ELWC tfrecords: padding so that each cells list has the same size, needed for tf-ranking
        r_cv.create_ELWC_tfrecord(contexts_train, filename=path_train,
                             padding=list_size, cell_wise=cell_wise)

        print("writing test record")
        r_cv.create_ELWC_tfrecord(contexts_test, filename=path_test, 
                             padding=list_size, cell_wise=cell_wise)
        
        
        
        cur_model_dir = model_dir + '/' + str(filename) + '_' + str(split_nr)
        if os.path.isdir(cur_model_dir):
            file_list = os.listdir(cur_model_dir)
            for i in range(len(file_list)):
                try:
                    os.remove(cur_model_dir + '/' + file_list[i])
                except:
                    print('failed to delete ' + cur_model_dir + '/' + file_list[i])

        # run model
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
            np.save(np_save_path, predictions)
        
        
        
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
            out_df.to_csv(csv_save_path,sep=',')
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
            out_df.to_csv(csv_save_path,sep=',')
    
        
        if fold_nr != -1:
            break
        
if __name__ == "__main__":
    main()
