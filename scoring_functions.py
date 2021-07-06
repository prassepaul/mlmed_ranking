#!/usr/bin/env python
# coding: utf-8

# nn baseline scoring function
import tensorflow as tf

def nn_baseline_make_score_fn(context_feature_columns,
                              example_feature_columns,
                              cell_wise,
                              batch_size = 1, list_size=390,
                              hidden_layer_dims = ["64", "32", "16"],
                              dropout_rate=0.4, vocab_size=31, embed_size=16):

    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        if(cell_wise):
            with tf.compat.v1.name_scope("smiles_embedding"):
                tokens  = [
                  tf.compat.v1.layers.flatten(group_features[name])
                  for name in sorted(example_feature_columns())
                ]

                tokens = tf.concat(tokens, 1)

                group_input = embedding_layer(
                                            tokens, vocab_size=vocab_size,
                                            embed_size=embed_size,
                                            name='smiles_embedding'
                                        )
                group_input =  tf.compat.v1.layers.flatten(group_input)

            with tf.compat.v1.name_scope("input_layer"):
                context_input = [
                  tf.compat.v1.layers.flatten(context_features[name])
                  for name in sorted(context_feature_columns())
                ]
                input_layer = tf.concat(context_input + [group_input], 1)

        else:
            with tf.compat.v1.name_scope("smiles_embedding"):
                tokens = [
                  tf.compat.v1.layers.flatten(context_features[name])
                  for name in sorted(context_feature_columns())
                ]

                tokens = tf.concat(tokens, 1)
                
                context_input = embedding_layer(
                                            tokens, vocab_size=vocab_size,
                                            embed_size=embed_size,
                                            name='smiles_embedding'
                                        )
                context_input =  tf.compat.v1.layers.flatten(context_input)

            with tf.compat.v1.name_scope("input_layer"):
                group_input = [
                  tf.compat.v1.layers.flatten(group_features[name])
                  for name in sorted(example_feature_columns())
                ]
                input_layer = tf.concat([context_input] + group_input, 1)


        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = input_layer
        cur_layer = tf.compat.v1.layers.batch_normalization(
          cur_layer,
          training=is_training,
          momentum=0.99)

        for i, layer_width in enumerate(int(d) for d in hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
            cur_layer,
            training=is_training,
            momentum=0.99)
            cur_layer = tf.nn.relu(cur_layer)
            cur_layer = tf.compat.v1.layers.dropout(
              inputs=cur_layer, rate=dropout_rate, training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn








# paccmann scoring function
from paccmann_custom_layers import (sequence_attention_layer,
                                    dense_attention_layer,
                                    embedding_layer,
                                    contextual_attention_layer,
                                    contextual_attention_matrix_layer)

def paccmann_make_score_fn(batch_size = 1, list_size=390, cell_wise=True,
                            smiles_vocabulary_size = 28,
                            other_output = None):
  

  """Returns a scoring function to build `EstimatorSpec`."""
  def mca_fn(features, mode, params):
    """
    Implement model for IC50 prediction based on selected genes attention and
    a multiscale attentive cnn.

    Args:
        - features: features for the observations (<dict<string, tf.Tensor>>).
        - labels: labels associated (<tf.Tensor>).
        - mode: mode for the model (<tf.estimator.ModeKeys>).
        - params: parameters for the model (<dict<string, object>>).
            Mandatory parameters are:
                - selected_genes_name: name of the selected genes features
                    (<string>).
                - tokens_name: name of the tokens features (<string>).
                - smiles_embedding_size: dimension of tokens' embedding
                    (<int>).
                - smiles_vocabulary_size: size of the tokens vocabulary
                    (<int>).

            Optional parameters for the model:
                - filters: numbers of filters to learn per convolutional 
                    layer (<list<int>>).
                - kernel_sizes: xizes of kernels per convolutional layer
                    (<list<list<int>>>).
                - multiheads: amount of attentive multiheads per SMILES
                    embedding. (<list<int>>). Should have len(filters)+1
                - stacked_dense_hidden_sizes:  sizes of the hidden dense
                    layers (<list<int>>).
                - smiles_attention: type of attention to be applied on encoded
                    smiles. Default: None. <string> in
                    {"sequence", "contextual", "matrix"}.
                - smiles_attention_size: size of the attentive layer for the
                    smiles sequence (<int>).
                - smiles_reduction: whether time dimension of post-cnn
                    attention is reduced (<bool>). Defaults to True. 
                    Does not apply for matrix attention.
                NOTE: The kernel sizes should match the dimensionality of the
                            smiles_embedding_size, so if the latter is 8, the
                            images are sequence_length x 8, then treat the 8
                            embedding dimensions like channels in an RGB image.

            Example params:
            ```
            {  
                "selected_genes_name": "query_features",
                "tokens_name": "document_features",
                "smiles_attention":  true,
                "smiles_attention_size": 8,
                "smiles_vocabulary_size": 28,
                "smiles_embedding_size": 8,
                "filters": [128, 128],
                "kernel_sizes": [[3, 8], [5, 8]],
                "multiheads":[32, 32, 32]
                "stacked_dense_hidden_sizes": [512, 64, 16]
            }
            ```
    Returns:
        The predictions in the form of a 1D `tf.Tensor` and a prediction
        dictionary (<dict<string, tf.Tensor>>).
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

#     # For de-standardization of the IC50 prediction.
#     min_ic50 = params.get('min', 0.0)
#     max_ic50 = params.get('max', 0.0)

    dropout = params.get('dropout', 0.0) if is_training else 0.0
    batch_size = (
        params['batch_size']
        if is_training else params['eval_batch_size']
    )
    
    token_str = "group_features" if cell_wise else "query_features"
    gene_str = "query_features" if cell_wise else "group_features"

    tokens = features[token_str][params['tokens_name']]
    sequence_length = tokens.shape[1]
    
    # Use transcriptomics and genomics
    if (
        params.get('use_cnv_data', False) and
        params.get('use_gep_data', True)
    ):
        # Genes will be of shape
        # `[batch_size, num_cnv_features + gep (5), num_genes (2128)]`.
        genes = assemble_cnv_gep_data(
           features, features[gene_str][params['selected_genes_name']]
        )
    # Use only transcriptomics.
    elif params.get('use_gep_data', True):
        genes = features[gene_str][params['selected_genes_name']] 
    # Use only genomics.
    elif params.get('use_cnv_data', False):
        genes = assemble_cnv_gep_data(features)
    num_gene_features = 1 if len(genes.shape) == 2 else genes.shape[2].value

    activation_fn = tf.nn.relu

    def attention_list_to_matrix(coding_tuple, axis=2):
        """ 
        Unpack the attention weights.

        Args:
            - coding_tuple: a list of tuples (outputs, att_weights) 
                coming from the attention function.
            - axis: the dimension along which expansion takes place 
                to concatenate the attention weights.
        
        Returns:
            - raw_coeff: a `tf.Tensor` with the attention weights of all 
                multiheads and convolutional kernel sizes concatenated
                along last dimension.
            - coeff: a `tf.Tensor` with the attention weights averaged
                along the given axis.
        """
        raw_coeff = tf.concat(
            [tf.expand_dims(t[1], 2) for t in coding_tuple], axis=axis
        )
        coeff =  tf.reduce_mean(raw_coeff, axis=axis)
        return raw_coeff, coeff


    # NOTE: tokens.shape[1].value = sequence_length = embedding_size.
    embedded_tokens = embedding_layer(
        tokens, params['smiles_vocabulary_size'],
        params['smiles_embedding_size'],
        name='smiles_embedding'
    )

    filters = params.get('filters', [32, 32])
    kernel_sizes = params.get(
        'kernel_sizes',
        [
            [3, params['smiles_embedding_size']],
            [5, params['smiles_embedding_size']]
        ]
    )
    multiheads = params.get('multiheads', [16, 16, 16])
    assert len(filters) == len(kernel_sizes)
    assert len(filters)+1 == len(multiheads)

    if params.get('dense_attention', False) == False:
        # If no dense attention is applied on genes, the same, unfiltered
        # genes are given as context to every contextual layer.
        encoded_genes = [genes]*len(multiheads)
        gene_attention_coefficients = tf.zeros(
            [batch_size, genes.shape[1].value]
        )
    elif params.get('gene_multihead', False) == False:
        # Dense attention is applied, but only ones, i.e. the same context.
        encoded_genes, gene_attention_coefficients = (
            dense_attention_layer(
                genes, return_alphas=True, name='gene_attention'
            )
        )
        encoded_genes = [encoded_genes]*len(multiheads)
    elif params.get('gene_multihead', False):
        # Filter genes differently for each SMILES kernel size.
        gene_tuple = [
            dense_attention_layer(
                genes, return_alphas=True, 
                name='gene_attention_{}'.format(l)
            ) for l in range(len(multiheads))
        ]
        encoded_genes = [tpl[0] for tpl in gene_tuple]
        gene_attention_coefficients_multi, gene_attention_coefficients = (
            attention_list_to_matrix(gene_tuple, axis=2)
        )

    # NOTE: Treat the sequence embedding matrix as an image.
    # Apply batch norm after activation function.
    def pad_sequence(data, kernel_size):
        """ 
        Pad the sequence.

        Args:
            - data: a `tf.Tensor` of shape .
            - axis: The dimension along which expansion takes place 
                to concatenate the attention weights.
        
        Returns:
            - raw_coeff: a `tf.Tensor` with the attention weights of all 
                multiheads and convolutional kernel sizes concatenated
                along last dimension.
            - coeff: a `tf.Tensor` with the attention weights averaged
                along the given axis.
        """
   
        
        pad = tf.expand_dims(
            embedding_layer(
                tf.zeros([batch_size, 1], dtype=tf.int32),
                params['smiles_vocabulary_size'],
                params['smiles_embedding_size']
            ), axis=3, name='smiles_padding'
        )
        pad_size = kernel_size[0] // 2
        return tf.concat([pad]*pad_size + [data] + [pad]*pad_size, axis=1)

    inputs = tf.expand_dims(embedded_tokens, 3)
    # i-th element has shape `[batch_size, T, filters(i)]`.
    convolved_smiles = [
        tf.compat.v1.layers.batch_normalization(
             tf.compat.v1.layers.dropout(
                tf.squeeze(
                     tf.compat.v1.layers.conv2d(
                        inputs=pad_sequence(inputs, kernel_size),
                        filters=num_kernel, kernel_size=kernel_size,
                        padding='valid', activation=activation_fn,
                        name='conv_{}'.format(index)
                    ),  axis=2
                ), rate=dropout
            ), training=is_training
        ) for index, (num_kernel, kernel_size) in enumerate(
            zip(filters, kernel_sizes)
        )
    ]
    # Complement convolved smiles with residual connection.
    convolved_smiles.insert(0, embedded_tokens)
    
    # Attention mechanism.
    if params.get('smiles_attention', None) == 'sequence':
        encoding_coefficient_tuple = [
            sequence_attention_layer(
                convolved_smiles[layer],
                params.get('smiles_attention_size', 256), return_alphas=True,
                reduce_sequence=params.get('smiles_reduction', True),
                name='sequence_attention_{}'.format(layer)
            ) for layer in range(len(convolved_smiles))
            for ind in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) == 'contextual':
        encoding_coefficient_tuple = [
            contextual_attention_layer(
                encoded_genes[layer], convolved_smiles[layer],
                params.get('smiles_attention_size', 256), return_alphas=True,
                reduce_sequence=params.get('smiles_reduction', True),
                name='contextual_attention_{}'.format(layer)
            ) for layer in range(len(convolved_smiles))
            for _ in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) == 'matrix':
        encoding_coefficient_tuple = [
            contextual_attention_matrix_layer(
                genes, convolved_smiles[layer], return_scores=True
            ) for layer in range(len(convolved_smiles))
            for _ in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) is not None:
        raise RuntimeError(
            'Unknown attention mechanism specified. Choose from '
            "{'sequence', 'contextual', 'matrix', None}."
        )

    # Done with attention, now prepare for concatenation with genes.
    # Check need to unpack list of tuples into encoded_smiles +
    # attention weights.
    if params.get('smiles_attention', None) is not None :
        if params.get('smiles_attention', None) == 'matrix':
            # Deal with attention weights first.
            # Each list entry of the tuple is of shape
            # `[batch_size, num_gene_features, sequence_length]`.
            attention_coefficients_raw, attention_coefficients = (
                attention_list_to_matrix(
                    encoding_coefficient_tuple, axis=3
                )
            )
            
            # Each output is shaped
            # `[batch_size, smiles_embedding_size, num_gene_features]`.
            encoded_smiles_list = [t[0] for t in encoding_coefficient_tuple]
            encoded_smiles = tf.concat(
                encoded_smiles_list, axis=1, name='encoded_smiles'
            )
            encoded_smiles.set_shape([
                batch_size, 
                (params['smiles_embedding_size']+num_gene_features) * 
                multiheads[0]+sum(
                    [
                        a*(b+num_gene_features)
                        for a, b in zip(multiheads[1:], filters)
                    ]
                )            
            ])
        # Applies for sequence or contextual attention
        else: 
            # Each alpha of the list of tuples is of shape
            # `[batch_size, sequence_length]`.
            # a_c_raw are of shape `[batch_size, T, multiheads * len(filters)]`
            # attention_coefficients is simply of shape `[batch_size, T]`.
            attention_coefficients_raw, attention_coefficients = (
                attention_list_to_matrix(
                    encoding_coefficient_tuple, axis=2
                )
            )
            encoded_smiles_list = [t[0] for t in encoding_coefficient_tuple]
            if params.get('smiles_reduction', True):
                # encoded_smiles is list of Tensors shape
                # `[batch_size, attention_size]`.
                encoded_smiles = tf.concat(
                        encoded_smiles_list, axis=1, name='encoded_smiles'
                )
                encoded_smiles.set_shape([
                    batch_size,
                    params['smiles_embedding_size']*multiheads[0] + 
                        sum([a * b for a, b in zip(multiheads[1:], filters)])
                ])

            else:
                # encoded_smiles is list of 3D Tensors of shape
                # `[batch_size, sequence_length, attention_size]`.
                encoded_smiles = [
                    tf.reshape(
                        encoded_smiles_list[layer],
                        [-1, sequence_length*filters[layer-1]]
                    ) for layer in range(1, len(encoded_smiles_list))
                ]
                encoded_smiles.insert(0, tf.reshape(
                    encoded_smiles_list[0],
                    [-1, sequence_length*params['smiles_embedding_size']]
                    )
                )
                encoded_smiles = tf.concat(
                    encoded_smiles, axis=1, name='encoded_smiles'
                )
                encoded_smiles.set_shape([
                    batch_size,
                    sequence_length * (
                        params['smiles_embedding_size']*multiheads[0]+ 
                        sum([a * b for a, b in zip(multiheads[1:], filters)])
                    )
                ])
    # In case no attention was applied
    else:
        encoded_smiles = [
            tf.reshape(
                convolved_smiles[layer+1],
                [-1, sequence_length*filters[layer]]
            ) for layer in range(len(convolved_smiles)-1)
        ]
        encoded_smiles.insert(0, tf.reshape(
                convolved_smiles[0],
                [-1, sequence_length*params['smiles_embedding_size']]
            )
        )

        encoded_smiles = tf.concat(
            encoded_smiles, axis=1, name='encoded_smiles'
        )

    # Apply batch normalization if specified
    layer = (
         tf.compat.v1.layers.batch_normalization(encoded_smiles, training=is_training)
        if params.get('batch_norm', False) else encoded_smiles
    )

    # NOTE: stacking dense layers as a bottleneck
    for index, dense_hidden_size in enumerate(
        params.get('stacked_dense_hidden_sizes', [])
    ):
        if not params.get('batch_norm', False):
            layer =  tf.compat.v1.layers.dropout(
                 tf.compat.v1.layers.dense(
                    layer, dense_hidden_size, activation=activation_fn,
                    name='dense_hidden_{}'.format(index)
                ),
                rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )
        # If batch_norm = True, look at position argument
        elif params.get('batch_norm_bef', True):
            layer =  tf.compat.v1.layers.dropout(
                activation_fn(
                     tf.compat.v1.layers.batch_normalization(
                         tf.compat.v1.layers.dense(
                            layer, dense_hidden_size,
                            activation=None,
                            name='dense_hidden_{}'.format(index)
                        ),
                        training=is_training,
                        name='batch_normed_dense_{}'.format(index)
                    ),
                    name='ouputs_dense_{}'.format(index)
                ),
                rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )
        # Then, batch_norm is applied after activation
        else:
            layer =  tf.compat.v1.layers.dropout(
                 tf.compat.v1.layers.batch_normalization(
                     tf.compat.v1.layers.dense(
                        layer, dense_hidden_size, activation=activation_fn,
                        name='outputs_dense_{}'.format(index)
                    ),
                    training=is_training,
                    name='batch_normed_dense_{}'.format(index)
                ), rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )

    predictions = tf.squeeze( tf.compat.v1.layers.dense(
        layer, 1, name='logits'
    ))
    prediction_dict = {
        'gene_attention': gene_attention_coefficients,
        'smiles_attention': attention_coefficients,
        'smiles_attention_raw': attention_coefficients_raw,
        'features': encoded_smiles
    }       
    # Converts IC50 to micromolar concentration if scaling
    # parameters available.
    # If unavailable, concentration will default to exp(0)=1.
    prediction_dict.update({
        'IC50': predictions
    })
    if params.get('gene_multihead', False):
        prediction_dict.update(
            {'gene_attention_raw': gene_attention_coefficients_multi}
        )
    
    if other_output is not None:
        predictions = prediction_dict[other_output]
        print(predictions.shape)
        print(type(predictions))
        #predictions = tf.compat.v1.expand_dims(predictions, axis=1)
        print(predictions.shape)
        #predictions = predictions[:,0]
        print(predictions.shape)
        return predictions
    else:
        predictions = tf.compat.v1.expand_dims(
        predictions, axis=1)

    return predictions

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""
    params = {  
    "batch_size": batch_size*list_size,
    "learning_rate": 0.0002,
    "dropout": 0.3,
    "batch_norm": True,
    "stacked_dense_hidden_sizes": [512, 128, 64, 16],
    "activation": "relu",
    "selected_genes_name": "query_features" if cell_wise else "document_features",
    "tokens_name": "document_features" if cell_wise else "query_features",
    "smiles_vocabulary_size": smiles_vocabulary_size,
    "smiles_embedding_size": 16,
    "multiheads":[4,4,4,4],
    "filters": [64,64,64],
    "kernel_sizes": [[3,16], [5,16], [11, 16]],
    "smiles_attention": "contextual",
    "smiles_attention_size": 64,
    "dense_attention":True,
    "gene_multihead": True,
    "buffer_size": 1000000,
    "prefetch_buffer_size": 512,
    "number_of_threads": 10,
    "drop_remainder": True,
    "eval_batch_size": batch_size*list_size}
  
    # remove the group dimension to adapt to paccmann scoring
    group_features["document_features"] = tf.squeeze(group_features["document_features"])
   
    return mca_fn({"query_features": context_features, "group_features": group_features}, mode=mode, params=params)

  return _score_fn

