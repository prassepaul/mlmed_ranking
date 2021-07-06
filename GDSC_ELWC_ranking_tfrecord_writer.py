#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from tensorflow_serving.apis import input_pb2
import os
import joblib
import pickle

flag_use_pickle = True # specify if you want to use pickle or joblib

tf.__version__ # I used 2.1.0



def create_feature_dict(data_df,
                        data_dir = 'data/gdsc_data/',
                        gene_feature = 'paccmann',
                        cell_wise = True):
    if not flag_use_pickle:
        cell_line_path = data_dir + 'cell_line_data.joblib'
        drug_path      = data_dir + 'drug_data.joblib'
        
        # load data
        cell_line_dict = joblib.load(cell_line_path)
        drug_dict = joblib.load(drug_path)  
    else:
        cell_line_path = data_dir + 'cell_line_data.pickle'
        drug_path      = data_dir + 'drug_data.pickle'
        
        with open(cell_line_path, 'rb') as handle:
            cell_line_dict = pickle.load(handle)
            
        with open(drug_path, 'rb') as handle:
            drug_dict = pickle.load(handle)
        
    
     
    
    
    cell_lines = cell_line_dict['cell_line_dict']
    drugs = drug_dict['drug_dict']
    
    # loop over dataframe
    data_matrix = np.array(data_df.to_numpy())
    num_not_nan = np.sum(~np.isnan(data_matrix))
    data_cell_lines = list(data_df.index)
    data_drugs = list(data_df.columns)
    counter = 0
    annotations = []
    drug_list = []
    cell_line_list = []
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i,j]):
                continue
            smiles_feature_vec  = drugs[data_drugs[j]]['feature_vec']
            gene_feature_vec    = cell_lines[str(data_cell_lines[i])][gene_feature + '_vector']
            if counter == 0:
                gene_features   = np.zeros([num_not_nan,len(gene_feature_vec)])
                smiles_features = np.zeros([num_not_nan,len(smiles_feature_vec)],dtype=np.int32)
                label           = np.zeros([num_not_nan,])
            gene_features[counter,:]   = gene_feature_vec
            smiles_features[counter,:] = smiles_feature_vec
            label[counter]             = data_matrix[i,j]
            annotations.append((data_cell_lines[i],data_drugs[j],data_drugs[j]))
            drug_list.append(data_drugs[j])
            cell_line_list.append(str(data_cell_lines[i]))
            
            counter += 1
    
    
    feature_dict={ 'selected_genes_20': gene_features,
                    'smiles_atom_tokens': smiles_features,
                    'label': label,
                    'drug_list':drug_list,
                    'cell_line_list':cell_line_list}
                    
    num_gene_features = gene_features.shape[1]
    num_smiles_features = smiles_features.shape[1]
    vocab_size = np.max(list(drug_dict['token_id_dict'].values()))
    return feature_dict, num_gene_features, num_smiles_features, vocab_size
    
    
def create_feature_dict_from_dicts(data_df,
                        cell_line_dict,
                        drug_dict,
                        gene_feature = 'paccmann',
                        cell_wise = True):
    
    cell_lines = cell_line_dict
    drugs = drug_dict
    
    # loop over dataframe
    data_matrix = np.array(data_df.to_numpy())
    num_not_nan = np.sum(~np.isnan(data_matrix))
    data_cell_lines = list(data_df.index)
    data_drugs = list(data_df.columns)
    counter = 0
    annotations = []
    drug_list = []
    cell_line_list = []
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i,j]):
                continue
            smiles_feature_vec  = drugs[data_drugs[j]]['feature_vec']
            gene_feature_vec    = cell_lines[str(data_cell_lines[i])][gene_feature + '_vector']
            if counter == 0:
                gene_features   = np.zeros([num_not_nan,len(gene_feature_vec)])
                smiles_features = np.zeros([num_not_nan,len(smiles_feature_vec)],dtype=np.int32)
                label           = np.zeros([num_not_nan,])
            gene_features[counter,:]   = gene_feature_vec
            smiles_features[counter,:] = smiles_feature_vec
            label[counter]             = data_matrix[i,j]
            annotations.append((data_cell_lines[i],data_drugs[j],data_drugs[j]))
            drug_list.append(data_drugs[j])
            cell_line_list.append(str(data_cell_lines[i]))
            
            counter += 1
    
    
    feature_dict={ 'selected_genes_20': gene_features,
                    'smiles_atom_tokens': smiles_features,
                    'label': label,
                    'drug_list':drug_list,
                    'cell_line_list':cell_line_list}
                    
    num_gene_features = gene_features.shape[1]
    num_smiles_features = smiles_features.shape[1]
    return feature_dict, num_gene_features, num_smiles_features


def create_context_dict(data_df,
                        data_dir = 'data/gdsc_data/',
                        gene_feature = 'paccmann',
                        cell_wise = True):
                        
    if not flag_use_pickle:
        cell_line_path = data_dir + 'cell_line_data.joblib'
        drug_path      = data_dir + 'drug_data.joblib'
        
        # load data
        cell_line_dict = joblib.load(cell_line_path)
        drug_dict = joblib.load(drug_path)  
    else:
        cell_line_path = data_dir + 'cell_line_data.pickle'
        drug_path      = data_dir + 'drug_data.pickle'
        
        with open(cell_line_path, 'rb') as handle:
            cell_line_dict = pickle.load(handle)
            
        with open(drug_path, 'rb') as handle:
            drug_dict = pickle.load(handle)
    
    
    cell_lines = cell_line_dict['cell_line_dict']
    drugs = drug_dict['drug_dict']
    
    # loop over dataframe
    data_matrix = np.array(data_df.to_numpy())
    num_not_nan = np.sum(~np.isnan(data_matrix))
    data_cell_lines = list(data_df.index)
    data_drugs = list(data_df.columns)
    counter = 0
    annotations = []
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i,j]):
                continue
            smiles_feature_vec  = drugs[data_drugs[j]]['feature_vec']
            gene_feature_vec    = cell_lines[str(data_cell_lines[i])][gene_feature + '_vector']
            if counter == 0:
                gene_features   = np.zeros([num_not_nan,len(gene_feature_vec)])
                smiles_features = np.zeros([num_not_nan,len(smiles_feature_vec)],dtype=np.int32)
                label           = np.zeros([num_not_nan,])
            gene_features[counter,:]   = gene_feature_vec
            smiles_features[counter,:] = smiles_feature_vec
            label[counter]             = data_matrix[i,j]
            annotations.append((data_cell_lines[i],data_drugs[j],data_drugs[j]))
            counter += 1
    
    
    context_dict = get_ELWC_dict(feature_dict={ 'selected_genes_20': gene_features,
                                                'smiles_atom_tokens': smiles_features,
                                                'label': label}, 
                    annotations=annotations, 
                    cell_wise=cell_wise)
    num_gene_features = gene_features.shape[1]
    num_smiles_features = smiles_features.shape[1]
    vocab_size = np.max(list(drug_dict['token_id_dict'].values()))
    return context_dict, num_gene_features, num_smiles_features, vocab_size
    
    
def create_context_dict_from_dicts(data_df,
                        cell_lines_dict,
                        drug_dict,
                        token_id_dict,
                        gene_feature = 'paccmann',
                        gene_appendix = '',
                        cell_wise = True): 
    
    
    cell_lines = cell_lines_dict
    drugs = drug_dict
    
    
    # loop over dataframe
    data_matrix = np.array(data_df.to_numpy())
    num_not_nan = np.sum(~np.isnan(data_matrix))
    data_cell_lines = list(data_df.index)
    data_drugs = list(data_df.columns)
    counter = 0
    annotations = []
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i,j]):
                continue
            smiles_feature_vec  = drugs[data_drugs[j]]['feature_vec']
            if str(data_cell_lines[i]) in cell_lines:
                gene_feature_vec    = cell_lines[str(data_cell_lines[i])][gene_feature + '_vector' + gene_appendix]
            elif data_cell_lines[i] in cell_lines:
                gene_feature_vec    = cell_lines[data_cell_lines[i]][gene_feature + '_vector' + gene_appendix]
            elif int(data_cell_lines[i]) in cell_lines:
                gene_feature_vec    = cell_lines[int(data_cell_lines[i])][gene_feature + '_vector' + gene_appendix]
            else:
                print('not found')
                print(allo)
            if counter == 0:
                gene_features   = np.zeros([num_not_nan,len(gene_feature_vec)])
                smiles_features = np.zeros([num_not_nan,len(smiles_feature_vec)],dtype=np.int32)
                label           = np.zeros([num_not_nan,])
            gene_features[counter,:]   = gene_feature_vec
            smiles_features[counter,:] = smiles_feature_vec
            label[counter]             = data_matrix[i,j]
            annotations.append((data_cell_lines[i],data_drugs[j],data_drugs[j]))
            counter += 1
    
    
    context_dict = get_ELWC_dict(feature_dict={ 'selected_genes_20': gene_features,
                                                'smiles_atom_tokens': smiles_features,
                                                'label': label}, 
                    annotations=annotations, 
                    cell_wise=cell_wise)
    num_gene_features = gene_features.shape[1]
    num_smiles_features = smiles_features.shape[1]
    vocab_size = np.max(list(token_id_dict.values()))
    return context_dict, num_gene_features, num_smiles_features, vocab_size


def get_context_dict(tfrecord_path = 'data/tfrecords/',
                    data_path='data/joined_paccmann_data/',
                    smiles_feature_path = 'data/joined_paccmann_data/smiles_atom_tokens.npy',
                    gene_feature_path = 'data/joined_paccmann_data/selected_genes_20.npy',
                    cell_wise=True
                    ):
    
    annotation_data_path = data_path + '/annotations.csv'
    label_data_path = data_path + '/ic50.npy'

    # cell and drug annotations of the drug-sensitivity experiments
    annotations= pd.read_csv(annotation_data_path)
    
    # IC50 values
    response = np.load(label_data_path)
    response = pd.Series(response)
    print("preprocessing")
    # get the features and filter out cell-drug pairs which were queried via the annotation_data but are not in the data
    features, annotations_filtered = get_features(data=annotations, label=response,
                                                smiles_feature_path=smiles_feature_path,
                                                gene_feature_path=gene_feature_path)

    
    # restructure the dicts to a list of drugs or cells (item) for each cell or drug (context) depending on cell_wise 
    context_dict = get_ELWC_dict(feature_dict=features, annotations=annotations_filtered, cell_wise=cell_wise)
    
    return context_dict


def get_features(data, label,
                smiles_feature_path = 'data/joined_paccmann_data/smiles_atom_tokens.npy',
                gene_feature_path = 'data/joined_paccmann_data/selected_genes_20.npy'):
    
    """
    function to get the features for the drugs and cells named in data from
    the paccmann data in smiles_feature_path (e.g. "data\\joined_paccmann_data\\smiles_atom_tokens.npy")

    
    Arguments: 
    data: pandas DataFrame with columns "cosmic_id", "inchi_key", the queried cell-drug experiments
    label: pandas Series, Series of ground the truth drug sensitivities of the experiments
    
    Returns: 
        a tuple of a feature dict with keys "selected_genes_20", "smiles_atom_tokens", "label" and 
        annotations of the dict
    
    
    """
    
    annotations = pd.read_csv("data/joined_paccmann_data/annotations.csv")
    all_drugs = annotations.inchi_key.unique()

    all_cells = annotations.cosmic_id.unique()

    # remove drugs and cells which are queried, but we do not have data for

    cells_in_query = data.cosmic_id.unique()
    drugs_in_query = data.inchi_key.unique()

    cells_not_in_data = set(cells_in_query)-set(all_cells)
    drugs_not_in_data = set(drugs_in_query)-set(all_drugs)
    if(cells_not_in_data):
        print("Removing " + str(len(cells_not_in_data)) + " of the queried cells, because missing  data. ")

        keep_row = np.array([True]*len(data))
        # find all rows of the data which relate to cells, for which we have no kernel data
        for cell in cells_not_in_data:
            keep_row = keep_row&(data.cosmic_id!= cell)
        data = data.loc[keep_row, :]
        label = label.loc[keep_row]

    if(drugs_not_in_data):
        print("Removing " + str(len(drugs_not_in_data)) + " of the queried drugs, because missing  data. ")

        keep_row = np.array([True]*len(data))
        # find all rows of the data which relate to drugs, for which we have no kernel data
        for drug in drugs_not_in_data:
            keep_row = keep_row&(data.inchi_key!= drug)
        data = data.loc[keep_row, :]
        label = label.loc[keep_row]


    cells_response = data.cosmic_id

    drugs_response = data.inchi_key

    # to filter out rows which are not queried
    mask_indices = []
    annotations_filtered =[]
    counter=0
    label = label.reset_index(drop=True)
    for drug, cell in tqdm(zip(drugs_response, cells_response), total=len(drugs_response)):

        is_experiment = (annotations.inchi_key==drug )& (annotations.cosmic_id == cell)
        if (np.any(is_experiment)):
            mask_indices.append(annotations[is_experiment].index[0])
            annotations_filtered.append(annotations[is_experiment].loc[:,["cosmic_id", "drug_names", "inchi_key"]].values[0])
        else: 
            label = label.drop(counter, axis=0)
        counter = counter+1

    smiles_atom_tokens = np.load(smiles_feature_path)[mask_indices,:]
    selected_genes_20 = np.load(gene_feature_path)[mask_indices,:]
    label = label.values.reshape(len(label) ,1) # DataFrame to reshaped numpy array
    

    annotations_filtered = np.stack(annotations_filtered)

    data_dict = {"selected_genes_20": selected_genes_20,
                     "smiles_atom_tokens": smiles_atom_tokens,
                    "label": label}
    return data_dict, annotations_filtered



def get_ELWC_dict(feature_dict, annotations, cell_wise=True):
    
        
    """
    function to create ELWC dict from a "linear" 
    feature dictionary with keys "selected_genes_20", "smiles_atom_tokens", "label"
    
    Arguments: 
    feature_dict: dict 
    annotations: pandas DataFrame: GDSC data annotations, returned as second argument from get_features
    cell_wise: Boolean: should the context be the cells or the drugs?
    
    Returns: 
    dict, example list with context nested dictionary:
    {"context_cell_genes": #array of the gene expression features of one cell-line,
                                      "examples":{"smiles_tokens": #2D array: n_drugs x n_tokens
                                      , "label": #array of label, "drug_name": #array of drug name}}
    
    """
    
    seen_context = set([])
    all_contexts = {}
    
    if cell_wise:
        for selec_g_20, smiles_a_t, label, annot in zip(feature_dict["selected_genes_20"],
                                                         feature_dict["smiles_atom_tokens"],
                                                         feature_dict["label"],
                                                        annotations):
            if(annot[0] in seen_context):
                all_contexts[annot[0]]["examples"]["smiles_tokens"].append(smiles_a_t)
                all_contexts[annot[0]]["examples"]["label"].append(label)
                all_contexts[annot[0]]["examples"]["drug_name"].append(annot[1])
            else:
                all_contexts[annot[0]] = {"context_cell_genes": selec_g_20,
                                          "examples":{"smiles_tokens": [smiles_a_t], "label": [label], "drug_name": [annot[1]]}}
                seen_context.add(annot[0])
    else:
        for selec_g_20, smiles_a_t, label, annot in zip(feature_dict["selected_genes_20"],
                                                 feature_dict["smiles_atom_tokens"],
                                                 feature_dict["label"],
                                                annotations):
            if(annot[2] in seen_context):
                all_contexts[annot[2]]["examples"]["selected_genes_20"].append(selec_g_20)
                all_contexts[annot[2]]["examples"]["label"].append(label)
                all_contexts[annot[2]]["examples"]["cell_name"].append(annot[0])
            else:
                all_contexts[annot[2]] = {"context_drug": smiles_a_t,
                                          "examples":{"selected_genes_20": [selec_g_20], "label": [label], "cell_name": [annot[0]]}}
                seen_context.add(annot[2])

            
    return all_contexts
        



# make sure gdsc data is in work_dir\\data
def create_ELWC_tfrecord(context_dict, filename, padding=-1, padding_rel=0, cell_wise=True):
    """
    function to create EWLC (Example list with context) tfrecord file for tensorflow-ranking
    from 2 or 3 features, depending on the presence of a label
    # the relevance is max(ic50)- ic50 since high ic50 means low relevance
    
    Arguments: 
    context_dict: example list with context dict as created by get_ELWC_dict()
    keys either "selected_genes_20", "smiles_atom_tokens" or "selected_genes_20", "smiles_atom_tokens", "label"
    
    filename: str, filename of the tfrecord which will be created
    
    padding: int, if -1 no padding will be applied, else examples will be added until the list of examples has length padding
    padding_rel: float, the relevance of the padded examples
    Returns: None
    
    """
    
    # helper functions for serialization
    def _float_feature(value_list):
        """Returns a float_list from a float / double."""
        if isinstance(value_list,list) or isinstance(value_list,np.ndarray):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))
        else:
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value_list]))

    def _int64_feature(value_list):
        """Returns an int64_list from a bool / enum / int / uint."""
        if isinstance(value_list,list) or isinstance(value_list,np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value_list]))
    

    CONTEXT = "context_cell_genes" if cell_wise else "context_drug"
    EXAMPLE = "smiles_tokens" if cell_wise else "selected_genes_20"

    with tf.io.TFRecordWriter(filename) as writer:
        for context in context_dict:
            
                if(len(context_dict[context]['examples']['label'])==0):
                    print("No drug experiments for cell id:")
                    print(context)
                    continue

                    
                context_specs = {}
                context_specs["query_features"] = _float_feature(context_dict[context][CONTEXT]) if cell_wise else _int64_feature(context_dict[context][CONTEXT])
                context_proto = tf.train.Example(features=tf.train.Features(feature=context_specs))


                ELWC = input_pb2.ExampleListWithContext()
                ELWC.context.CopyFrom(context_proto)
                
                # invert ic50 so that the max value has relevance zero and the lowest ic50 has the highest relevance
                # we do this since a high IC50 means low relevance/ineffective drug for the given cell context
                max_ic50 = np.max(context_dict[context]['examples']['label'])
                context_dict[context]['examples']['label'] = max_ic50 - context_dict[context]['examples']['label']
                
                n_examples = 0
                for doc, rel in zip(context_dict[context]['examples'][EXAMPLE], context_dict[context]['examples']['label']):

                    example_features = ELWC.examples.add()
                    example_specs = {}
                    example_specs["relevance"] = _float_feature(rel)
                    example_specs["document_features"] = _int64_feature(doc) if cell_wise else _float_feature(doc)
                    exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_specs))
                    example_features.CopyFrom(exampe_proto)
                    n_examples +=1
                    
                # add meaningless examples as padding (the lists of items for each context have to be the same size)    
                if(padding != -1): 
                    n_padding = padding - n_examples
                    #print('add ' + str(n_padding) + ' examples')
                    for _ in range(n_padding):
                        example_features = ELWC.examples.add()
                        example_specs = {}
                        example_specs["relevance"] = _float_feature([padding_rel])
                        example_specs["document_features"] = _int64_feature([1]*len(doc)) if cell_wise else _float_feature([-99]*len(doc))
                        exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_specs))
                        example_features.CopyFrom(exampe_proto)

                writer.write(ELWC.SerializeToString())



def cold_start_train_test_split(context_dict, eval_percentage=0.1, test_percentage=0.1, random_state=None):
    """
    function to train, eval, eval split a dictionary of contexts, as created by get_ELWC_dict()
    
    Arguments: 
    context_dict: example list with context dict as created by get_ELWC_dict()
    
    eval_percentage: float , percentage of the data used for evaluation
    test_percentage: float , percentage of the data used for testing

    random_state, int or None: random state for train_test_split
    
    Returns: (dict, dict, dict)
    
    """
    
    contexts = list(context_dict.keys())
    
    contexts_train, contexts_holdout = train_test_split(contexts, test_size=test_percentage+eval_percentage,
                                                        shuffle=True, random_state=random_state)
    contexts_eval, contexts_test = train_test_split(contexts_holdout, test_size=test_percentage/(test_percentage+eval_percentage),
                                                        shuffle=False)
    # create sub dictionaries with train/test/eval contexts
    context_dict_train ={k:context_dict[k] for k in contexts_train}
    context_dict_eval = {k:context_dict[k] for k in contexts_eval}
    context_dict_test = {k:context_dict[k] for k in contexts_test}
    
    # assert cold start: no context is part of more than one dict
    assert context_dict_train.keys() & context_dict_eval.keys() == set([])
    assert context_dict_eval.keys() & context_dict_test.keys() == set([])
    assert context_dict_train.keys() & context_dict_test.keys() == set([])
    
    return context_dict_train, context_dict_eval, context_dict_test

