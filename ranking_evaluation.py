#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import tensorflow as tf
try:
  from tensorflow_serving.apis import input_pb2
except ImportError:
  # This is needed for tensorboard compatibility.
  get_ipython().system('pip install -q tensorflow-serving-api ')
  from tensorflow_serving.apis import input_pb2



class RankingEvaluation():
    def __init__(self, predictions, test_tfrecord_path):
        """
        Class to evaluate relevance predictions
        
        predictions: list of lists of relevance scores
        test_tfrecord_path: str, path to test tfrecords which contain ground truth relevances
        """
        
        self.predictions = predictions
        
        filenames = [test_tfrecord_path]
        raw_dataset = tf.data.TFRecordDataset(filenames)

        ground_truth = []
        for e in raw_dataset.take(1000): 
            ELWC = input_pb2.ExampleListWithContext()
            v = ELWC.FromString(e.numpy())
            ground_truth_one_context = []
            for e in v.examples:
                ground_truth_one_context.append(e.features.feature["relevance"].float_list.value[0])

            ground_truth.append(ground_truth_one_context)
            
        self.ground_truth = ground_truth
    
    def ndcg(self, k=5, exponential_scaling=False):
        """
        see sklearn.metrics.ndcg for details
        ndcg is a list of ndcgs for every context (cell)
        """
        ndcg  =[]

        for example_rels, example_preds in zip(self.ground_truth, self.predictions):

            # two definitions of ndscs, exponential or not
            example_rels = np.exp(example_rels) if exponential_scaling else example_rels
            
            ndcg.append(ndcg_score([example_rels], [example_preds], k=k))

        return(ndcg)
    
    def prec_at_k(self, k = 5):
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

        for example_rels, example_preds in zip(self.ground_truth, self.predictions):
            example_rels = np.argsort(-np.array(example_rels ))[:k]
            example_preds = np.argsort(-np.array(example_preds))[:k]


            precs.append(precision(example_rels, example_preds, k=k))


        return(precs)




def random_ndsc_bl(example_rel_lists, k = None, n_samples = 100, exponential_scaling=False):
    random_rels = np.random.choice(range(390), (n_samples, 390))*1.

    ndcg  =[]
    for example_rels in example_rel_lists:
        y_true = [np.exp(example_rels)] if exponential_scaling else [example_rels]
        ndcg_samples =[]
        for random_rel in random_rels:

            # pad with zero rel for padded docs
            first_zero_rel = np.where(y_true)[1][-1]
            random_rel = np.concatenate([random_rel[:first_zero_rel], [0.]*(390-first_zero_rel)])

            ndcg_samples.append(ndcg_score(y_true, [random_rel], k=k))

        ndcg.append(np.mean(ndcg_samples))
    return(ndcg)

