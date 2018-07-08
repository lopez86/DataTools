import tensorflow as tf

from ..tf.model_layers import (
    concatenate, dense_layer, dropout, global_max_pool, simple_text_inputs,
    text_conv_layer, text_embedding_layer,
)


def binary_text_classification_cnn(
        sentence_length,
        vocab_size,
        embedding_size,
        convolutions,
        dense_size,
        do_keep_frac=0.7,
        optimizer=tf.train.AdamOptimizer()
):
    graph = tf.Graph()
    with graph.as_default():
        x, y, is_train = simple_text_inputs(sentence_length, 1)
        embeddings = text_embedding_layer(x, vocab_size, embedding_size)
        max_pools = []
        for conv_size, n_filters in convolutions.items():
            scope = 'conv{}'.format(conv_size)
            activations = text_conv_layer(
                embeddings, conv_size, n_filters,
                name_scope=scope, var_scope=scope
            )
            max_pools.append(global_max_pool(activations, scope))
        concatenated = concatenate(max_pools, name_scope='dense')
        dropped_out = dropout(
            concatenated, is_train, name_scope='dense', keep_prob=do_keep_frac
        )
        activations = dense_layer(
            dropped_out, dense_size,
            name_scope='dense', variable_scope='dense'
        )
        output = dense_layer(
            activations, 1, name_scope='output', variable_scope='output',
            activation=tf.nn.sigmoid
        )
        with tf.name_scope('output'):
            predictions = tf.identity(output, name='predictions')
            loss = tf.identity(
                tf.losses.log_loss(y, predictions),
                name='loss'
            )
            training_op = optimizer.minimize(loss, name='training')

    return graph