from scoring_functions import nn_baseline_make_score_fn, paccmann_make_score_fn

import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import input_pb2
import os
import tensorflow_ranking as tfr

    
    
def eval_metric_fns():
    """Returns a dict from name to metric functions.

    Returns:
    A dict mapping from metric name to a metric function with above signature.
    """

    metric_fns = {}
    metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10, 390]
    })

    return metric_fns


class Model():
    def __init__(self,
                scoring="paccmann",
                loss="mse",
                model_dir="ranking_model_dir",
                padding_label=0,
                label_feature="relevance",
                n_context_feature=2128,
                n_example_feature=155,
                list_size=390,
                cell_wise=True,
                smiles_vocabulary_size = 28):
        
        """
        model function for a GDSC ranking model

        scoring: str, type of scoring function to use "paccmann" or "nn_baseline"
        loss: str, type of loss to use "approx_ndcg" or "mse"
        padding_label: int, padding label for shorter context lists (usually should be 0 so that those values are ignored)
        label_feature, str, name of the label feature
        n_context_feature: int, number of cell features
        n_example feature: int, number of drug features
        list_size: size of the longest example list, all other lists are padded to that size
        model_dir: path were trained model is stored
        """
        
        self.n_context_feature = n_context_feature
        self.n_example_feature = n_example_feature
        self.padding_label = padding_label
        self.list_size = list_size
        self.label_feature = label_feature
        self.model_dir = model_dir
        self.scoring = scoring
        self.batch_size = 1 # currently only batch size of 1 possible due to ranking
        self.cell_wise = cell_wise
        self.smiles_vocabulary_size = smiles_vocabulary_size
        
        # define the loss fn
        if (loss == "mse"):
            loss = tfr.losses.RankingLossKey.MEAN_SQUARED_LOSS
        elif(loss == "approx_ndcg"):
            loss = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
        else:
            raise(NotImplementedError("loss has to be 'approx_ndcg' or 'mse'"))

        self.loss_fn = tfr.losses.make_loss_fn(loss)



    def train(self,
              learning_rate=0.05, 
              num_train_steps = 15 * 10000,
              train_data_path='data/tfrecords/train.tfrecord',
              eval_data_path=None):
        
        
        # eval data is currently just used for verbose during training
        if(eval_data_path is None):
            eval_data_path = train_data_path
        
        
        
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)

        def context_feature_columns():
            """Returns context feature names to column definitions."""
            dtype=tf.dtypes.float32 if self.cell_wise else tf.dtypes.int64
            
            context_feature_column = tf.feature_column.numeric_column(
            "query_features", shape= (self.n_context_feature), default_value=None, dtype=dtype
            )

            return {"query_features": context_feature_column}

        
        
        
        def example_feature_columns():
            
            dtype= tf.dtypes.int64 if self.cell_wise else tf.dtypes.float32
            
            example_feature_column = tf.feature_column.numeric_column(
            "document_features", shape=(self.n_example_feature), default_value=None, dtype=dtype
            )
            return {"document_features": example_feature_column}

        
        

        def make_transform_fn():
            def _transform_fn(features, mode):
                """Defines transform_fn."""
                context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=context_feature_columns(),
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")
                return context_features, example_features
            return _transform_fn
        
        
        def _train_op_fn(loss):
            """Defines train op used in ranking head."""
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            minimize_op = optimizer.minimize(
              loss=loss, global_step=tf.compat.v1.train.get_global_step())
            train_op = tf.group([update_ops, minimize_op])
            return train_op

        ranking_head = tfr.head.create_ranking_head(
              loss_fn=self.loss_fn,
              eval_metric_fns=eval_metric_fns(),
              train_op_fn=_train_op_fn)
        
        
        
        # define the neural network
        if(self.scoring == "paccmann"):
            self.model_fn = tfr.model.make_groupwise_ranking_fn(
                      group_score_fn=paccmann_make_score_fn(list_size=self.list_size, cell_wise=self.cell_wise,
                                            smiles_vocabulary_size = self.smiles_vocabulary_size),
                      transform_fn=make_transform_fn(),
                      group_size=1,
                      ranking_head=ranking_head)
        elif(self.scoring == "nn_baseline"):
            self.model_fn = tfr.model.make_groupwise_ranking_fn(
                      group_score_fn=nn_baseline_make_score_fn(
                                                            hidden_layer_dims = ["64", "32", "16"],
                                                            dropout_rate=0.4, 
                                                            context_feature_columns=context_feature_columns,
                                                            example_feature_columns=example_feature_columns,
                                                            list_size=self.list_size, cell_wise=self.cell_wise),
                      transform_fn= make_transform_fn(),
                      group_size=1,
                      ranking_head=ranking_head)
        else:
            raise(NotImplementedError("scoring has to be 'paccmann' or 'nn_baseline'"))
            
            
            
            
        # data input function
        def input_fn(path, num_epochs=None):
            if self.cell_wise:
                context_feature_column = tf.feature_column.numeric_column(
                "query_features", shape=(self.n_context_feature), default_value=-0., dtype=tf.dtypes.float32
                )
            else:
                context_feature_column = tf.feature_column.numeric_column(
                "query_features", shape=(self.n_context_feature), default_value=0, dtype=tf.dtypes.int64
                )
            context_feature_spec = tf.feature_column.make_parse_example_spec(
                [context_feature_column])

            label_column = tf.feature_column.numeric_column(
                self.label_feature, dtype=tf.float32, default_value=self.padding_label)
            if self.cell_wise:
                example_feature_column = tf.feature_column.numeric_column(
                "document_features", shape=(self.n_example_feature), default_value=0, dtype=tf.dtypes.int64
                )
            else:
                example_feature_column = tf.feature_column.numeric_column(
                "document_features", shape=(self.n_example_feature), default_value=-0., dtype=tf.dtypes.float32
                )
            example_feature_spec = tf.feature_column.make_parse_example_spec(
                [example_feature_column, label_column])
            dataset = tfr.data.build_ranking_dataset(
                file_pattern=path,
                data_format=tfr.data.ELWC,
                batch_size=self.batch_size,
                list_size=self.list_size,
                context_feature_spec=context_feature_spec,
                example_feature_spec=example_feature_spec,
                reader=tf.data.TFRecordDataset,
                shuffle=False,
                num_epochs=num_epochs )

            features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

            label = tf.squeeze(features.pop(self.label_feature), axis=2)
            label = tf.cast(label, tf.float32)
            return features, label
        
        def train_and_eval_fn():
            
            """Train and eval function used by `tf.estimator.train_and_evaluate`."""
            run_config = tf.estimator.RunConfig(
              save_checkpoints_steps=5000)
            ranker = tf.estimator.Estimator(
              model_fn=self.model_fn,
              model_dir=self.model_dir,
              config=run_config)

            train_input_fn = lambda: input_fn(train_data_path)
            eval_input_fn = lambda: input_fn(eval_data_path)#,num_epochs=1)

            train_spec = tf.estimator.TrainSpec(
              input_fn=train_input_fn, max_steps=num_train_steps)
            eval_spec =  tf.estimator.EvalSpec(
                  name="eval",
                  input_fn=eval_input_fn,
                  throttle_secs=15)
            return (ranker, train_spec, eval_spec)
        
        ranker, train_spec, eval_spec = train_and_eval_fn()
        tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
        self.ranker = ranker
        
        
        
    def predict(self, test_size,
                  test_data_path = 'data/tfrecords/test.tfrecord'):
        
        
        def context_feature_columns():
            """Returns context feature names to column definitions."""
            dtype=tf.dtypes.float32 if self.cell_wise else tf.dtypes.int64
            
            context_feature_column = tf.feature_column.numeric_column(
            "query_features", shape= (self.n_context_feature), default_value=None, dtype=dtype
            )

            return {"query_features": context_feature_column}

        
        
        
        def example_feature_columns():
            
            dtype= tf.dtypes.int64 if self.cell_wise else tf.dtypes.float32
            
            example_feature_column = tf.feature_column.numeric_column(
            "document_features", shape=(self.n_example_feature), default_value=None, dtype=dtype
            )
            return {"document_features": example_feature_column}

        
        

        def make_transform_fn():
            def _transform_fn(features, mode):
                """Defines transform_fn."""
                context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=context_feature_columns(),
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")
                return context_features, example_features
            return _transform_fn
        
        
        if not hasattr(self,'ranker'):        
            # define the neural network
            
            # create dummy optimizer
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.03)           
            
            ranking_head = tfr.head.create_ranking_head(
                  loss_fn=self.loss_fn)            
            
            if(self.scoring == "paccmann"):
                self.model_fn = tfr.model.make_groupwise_ranking_fn(
                          group_score_fn=paccmann_make_score_fn(list_size=self.list_size, cell_wise=self.cell_wise,
                                                smiles_vocabulary_size = self.smiles_vocabulary_size),
                          transform_fn=make_transform_fn(),
                          group_size=1,
                          ranking_head=ranking_head)
            elif(self.scoring == "nn_baseline"):
                self.model_fn = tfr.model.make_groupwise_ranking_fn(
                          group_score_fn=nn_baseline_make_score_fn(
                                                                hidden_layer_dims = ["64", "32", "16"],
                                                                dropout_rate=0.4, 
                                                                context_feature_columns=context_feature_columns,
                                                                example_feature_columns=example_feature_columns,
                                                                list_size=self.list_size, cell_wise=self.cell_wise),
                          transform_fn= make_transform_fn(),
                          group_size=1,
                          ranking_head=ranking_head)
            
        
            run_config = tf.estimator.RunConfig(
              save_checkpoints_steps=5000)
            self.ranker = tf.estimator.Estimator(
              model_fn=self.model_fn,
              model_dir=self.model_dir,
              config=run_config)

        
        """
        predict the ranking relevances for a test tfrecord dataset at test_data_path
        using the model in the ranking model dir,
        
        returns the predictions
        
        test_size: int, number of test examples
        test_data_path: str, path to the test examples
        """
        
        def predict_input_fn(path, num_epochs=None):
            dtype_context  = tf.dtypes.float32 if self.cell_wise else tf.dtypes.int64
            default_context = 0. if self.cell_wise else 0
            dtype_examples = tf.dtypes.int64 if self.cell_wise else tf.dtypes.float32
            default_examples = 0 if self.cell_wise else 0.

            context_feature_column = tf.feature_column.numeric_column(
            "query_features", shape=(self.n_context_feature), default_value=default_context, dtype=dtype_context
            )
            context_feature_spec = tf.feature_column.make_parse_example_spec(
                [context_feature_column])

            label_column = tf.feature_column.numeric_column(
                self.label_feature, dtype=tf.float32, default_value=self.padding_label)

            example_feature_column = tf.feature_column.numeric_column(
            "document_features", shape=(self.n_example_feature), default_value=default_examples, dtype=dtype_examples
            )
            example_feature_spec = tf.feature_column.make_parse_example_spec(
                [example_feature_column, label_column])
            dataset = tfr.data.build_ranking_dataset(
                file_pattern=path,
                data_format=tfr.data.ELWC,
                batch_size=1,
                list_size=self.list_size,
                context_feature_spec=context_feature_spec,
                example_feature_spec=example_feature_spec,
                reader=tf.data.TFRecordDataset,
                shuffle=False,
                num_epochs=num_epochs)

            features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
            
            print(features.keys())
            print(self.label_feature)
            
            label = tf.squeeze(features.pop(self.label_feature), axis=2)
            return features


        predictions = self.ranker.predict(input_fn=lambda: predict_input_fn(test_data_path))
        # get the predictions
        preds = []
        for _ in range(test_size):
            x = next(predictions)
            assert(len(x) == self.list_size) 
            preds.append(x)
        return(preds)
    
    
    
    def predict_intermediate(self, test_size,
                  test_data_path = 'data/tfrecords/test.tfrecord',
                  other_output = 'gene_attention'):
        
        
        def context_feature_columns():
            """Returns context feature names to column definitions."""
            dtype=tf.dtypes.float32 if self.cell_wise else tf.dtypes.int64
            
            context_feature_column = tf.feature_column.numeric_column(
            "query_features", shape= (self.n_context_feature), default_value=None, dtype=dtype
            )

            return {"query_features": context_feature_column}

        
        
        
        def example_feature_columns():
            
            dtype= tf.dtypes.int64 if self.cell_wise else tf.dtypes.float32
            
            example_feature_column = tf.feature_column.numeric_column(
            "document_features", shape=(self.n_example_feature), default_value=None, dtype=dtype
            )
            return {"document_features": example_feature_column}

        
        

        def make_transform_fn():
            def _transform_fn(features, mode):
                """Defines transform_fn."""
                context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=context_feature_columns(),
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")
                return context_features, example_features
            return _transform_fn
        
        
        if not hasattr(self,'ranker'):        
            # define the neural network
            
            # create dummy optimizer
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.03)           
            
            ranking_head = tfr.head.create_ranking_head(
                  loss_fn=self.loss_fn)            
            
            if(self.scoring == "paccmann"):
                self.model_fn = tfr.model.make_groupwise_ranking_fn(
                          group_score_fn=paccmann_make_score_fn(list_size=self.list_size, cell_wise=self.cell_wise,
                                                smiles_vocabulary_size = self.smiles_vocabulary_size,
                                                other_output = other_output),
                          transform_fn=make_transform_fn(),
                          group_size=1,
                          ranking_head=ranking_head)
            elif(self.scoring == "nn_baseline"):
                self.model_fn = tfr.model.make_groupwise_ranking_fn(
                          group_score_fn=nn_baseline_make_score_fn(
                                                                hidden_layer_dims = ["64", "32", "16"],
                                                                dropout_rate=0.4, 
                                                                context_feature_columns=context_feature_columns,
                                                                example_feature_columns=example_feature_columns,
                                                                list_size=self.list_size),
                          transform_fn= make_transform_fn(),
                          group_size=1,
                          ranking_head=ranking_head)
            
                        
            run_config = tf.estimator.RunConfig(
              save_checkpoints_steps=5000)
            self.ranker = tf.estimator.Estimator(
              model_fn=self.model_fn,
              model_dir=self.model_dir,
              config=run_config)

        
        """
        predict the ranking relevances for a test tfrecord dataset at test_data_path
        using the model in the ranking model dir,
        
        returns the predictions
        
        test_size: int, number of test examples
        test_data_path: str, path to the test examples
        """
        
        def predict_input_fn(path, num_epochs=None):
            dtype_context  = tf.dtypes.float32 if self.cell_wise else tf.dtypes.int64
            default_context = 0. if self.cell_wise else 0
            dtype_examples = tf.dtypes.int64 if self.cell_wise else tf.dtypes.float32
            default_examples = 0 if self.cell_wise else 0.

            context_feature_column = tf.feature_column.numeric_column(
            "query_features", shape=(self.n_context_feature), default_value=default_context, dtype=dtype_context
            )
            context_feature_spec = tf.feature_column.make_parse_example_spec(
                [context_feature_column])

            label_column = tf.feature_column.numeric_column(
                self.label_feature, dtype=tf.float32, default_value=self.padding_label)

            example_feature_column = tf.feature_column.numeric_column(
            "document_features", shape=(self.n_example_feature), default_value=default_examples, dtype=dtype_examples
            )
            example_feature_spec = tf.feature_column.make_parse_example_spec(
                [example_feature_column, label_column])
            dataset = tfr.data.build_ranking_dataset(
                file_pattern=path,
                data_format=tfr.data.ELWC,
                batch_size=1,
                list_size=self.list_size,
                context_feature_spec=context_feature_spec,
                example_feature_spec=example_feature_spec,
                reader=tf.data.TFRecordDataset,
                shuffle=False,
                num_epochs=num_epochs)

            features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
            
            print(features.keys())
            print(self.label_feature)
            
            label = tf.squeeze(features.pop(self.label_feature), axis=2)
            return features

        print('ranker: ' + str(self.ranker))
        #print('ranker var names: ' + str(self.ranker.get_variable_names()))
        predictions = self.ranker.predict(input_fn=lambda: predict_input_fn(test_data_path))
        #print('prediction_keys: ' + str(predictions_dict.keys()))
        print('predictions')
        print(predictions)
        print(next(predictions))
        print(allo)
        
        
        # get the predictions
        preds = []
        for _ in range(test_size):
            x = next(predictions)
            assert(len(x) == self.list_size) 
            preds.append(x)
        return(preds)
    
