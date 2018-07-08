import tensorflow as tf


def simple_text_inputs(n_feat, n_outputs, name_scope='inputs', sparse_features=False):
    with tf.name_scope(name_scope):
        if sparse_features:
            x = tf.sparse_placeholder(tf.float32, shape=(None, n_feat), name='x')
        else:
            x = tf.placeholder(tf.int32, shape=(None, n_feat), name='x')
        y = tf.placeholder(tf.float32, shape=(None, n_outputs), name='y')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

    return x, y, is_train


def text_embedding_layer(
        input_tensor,
        vocab_size,
        embedding_size,
        name_scope='embedding',
        var_scope='embedding',
        initializer=tf.contrib.layers.xavier_initializer()
):
    with tf.name_scope(name_scope):
        with tf.variable_scope(var_scope):
            embedding_vars = tf.get_variable(
                'embeddings',
                [vocab_size, embedding_size],
                dtype=tf.float32,
                initializer=initializer
            )

        embedded = tf.nn.embedding_lookup(embedding_vars, input_tensor)
    return embedded


def text_conv_layer(
        input_tensor,
        conv_size,
        n_filters,
        name_scope,
        var_scope,
        activation=tf.nn.relu,
        initializer=tf.contrib.layers.xavier_initializer()
):
    if len(input_tensor.shape) < 4:
        input_tensor = tf.reshape(
            input_tensor,
            [-1, input_tensor.shape[1].value, input_tensor.shape[2].value, 1]
        )

    with tf.name_scope(name_scope):
        with tf.variable_scope(var_scope):
            filters = tf.get_variable(
                'filters',
                [conv_size, input_tensor.shape[2].value, input_tensor.shape[3].value, n_filters],
                dtype=tf.float32,
                initializer=initializer
            )
            biases = tf.get_variable(
                'biases',
                [n_filters],
                initializer=tf.zeros_initializer()
            )
        convolved = tf.nn.conv2d(
            input_tensor,
            filter=filters,
            strides=[1, 1, 1, 1],
            padding='VALID'
        ) + biases
        activations = activation(
            convolved, name='activation'
        )
    return activations


def global_max_pool(input_tensor, name_scope):
    with tf.name_scope(name_scope):
        max_pool = tf.nn.max_pool(
            input_tensor,
            ksize=[1, input_tensor.shape[1].value, input_tensor.shape[2].value, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='global_max_pool'
        )
    return max_pool


def concatenate(tensor_list, name_scope):
    with tf.name_scope(name_scope):
        concatenated = tf.concat(
            [
                tf.reshape(tensor, [-1, tensor.shape[3].value])
                for tensor in tensor_list
            ],
            1, name='concatenated'
        )
    return concatenated


def dropout(tensor, is_train, keep_prob, name_scope):
    with tf.name_scope(name_scope):
        do = tf.cond(
            tf.equal(is_train, False),
            lambda: tensor,
            lambda: tf.nn.dropout(tensor, keep_prob=keep_prob)
        )
    return do


def dense_layer(
        tensor,
        n_units,
        name_scope,
        variable_scope,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation=tf.nn.relu,
        sparse_input=False
):
    with tf.name_scope(name_scope):
        with tf.variable_scope(variable_scope):
            w = tf.get_variable('weights', [tensor.shape[1].value, n_units], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable('biases', [n_units], dtype=tf.float32, initializer=initializer)
            if sparse_input:
                logits = tf.sparse_tensor_dense_matmul(tensor, w) + b
            else:
                logits = tf.matmul(tensor, w) + b
            if activation is not None:
                activations = activation(logits, name='activations')
            else:
                activations = tf.identity(logits, name='activations')
    return activations
