import numpy as np
import tensorflow as tf

from .feed_dict import simple_feed_builder
from ..batches import generate_batches
from ..data_types import Results


# This is in the form {tf.Graph variable name: greater_loss_is_better}
DEFAULT_LOSS_DICT = {'output/loss': False}
# This is in the form {output variable name: tf graph variable name}
DEFAULT_PRED_DICT = {'y': 'output/predictions'}


def train_model_with_stopping(
    model_builder,
    preprocessor,
    train_data,
    val_data=None,
    test_data=None,
    feed_builder=simple_feed_builder,
    batch_generator=generate_batches,
    n_stop=-1,
    verbose=1,
    losses=DEFAULT_LOSS_DICT,
    train_str='output/training',
    pred_dict=DEFAULT_PRED_DICT
):
    """Train a Tensorflow model with rudimentary early stopping.

    The preprocessor takes in train_data, val_data, and test_data
    and outputs (train, val, test, feature_names), where train, val, and test
    are Datasets. It does any additional processing that hasn't been done yet.

    The model_builder takes in the output of the preprocessor and outputs
    a tensorflow graph.

    The feed_builder takes in a batch and a boolean flag to designate whether
    we are on a training step or not and outputs a tensorflow feed dict.

    The batch_generator takes in a Dataset and outputs a BatchData instance.

    Args:
        model_builder: callable, builds the tensorflow graph
        preprocessor: callable,
        train_data: Dataset instance
        val_data: maybe(Dataset) instance
        test_data: maybe(Dataset) instance
        feed_builder: callable, creates a feed dict
        batch_generator: callable, generates batches
        n_stop: int, number of epochs with no improvement before stopping.
                If 0 or negative, the stopping is disabled.
        verbose: int, verbosity level
        losses: dict of str -> bool, keys are loss functions to check, values give
                whether the function is better at higher values or not.
        train_str: str, the training operation for the graph
        pred_dict: dict of str -> str, gives the mapping of output variable name to
                   tensorflow graph variable name.


    Returns:
        Results instance with predictions on any validation or test data.
    """
    best_scores, best_epochs = _initialize_stopping(losses)

    train, val, test, features = preprocessor(
        train_data, val_data, test_data
    )
    model = model_builder(train, val, test, features)
    val_feed_dict = feed_builder(val, False)

    with tf.Session(graph=model) as sess:
        tf.global_variables_initializer().run()

        for batch in batch_generator(train):
            train_feed_dict = feed_builder(batch, True)

            sess.run(train_str, feed_dict=train_feed_dict)
            if batch.epoch_done and val_feed_dict is not None:

                if verbose == 0 and n_stop <= 0:
                    continue

                scores = _get_validation_losses(
                    sess, losses.keys(), val_feed_dict, verbose
                )

                best_epochs, best_scores, stop_training = _check_early_stopping(
                    losses, scores, best_epochs, best_scores, batch.epoch, n_stop
                )

                if stop_training:
                    if verbose > 0:
                        print('Stopping training at epoch {}'.format(batch.epoch))
                    break

        val_preds = _run_predictions(sess, pred_dict, data=val, feed_dict=val_feed_dict)
        test_preds = _run_predictions(
            sess, pred_dict, data=test, feed_builder=feed_builder
        )

        results = Results(validation=val_preds, test=test_preds)
        return results


def _run_predictions(sess, pred_dict, feed_dict=None, data=None, feed_builder=None):
    """With the current active session, make predictions from a given dataset.

    A feed dict may be passed directly into this function but otherwise, a Dataset
    and a function to build a feed dict must be used instead.

    Args:
        sess: tensorflow.Session instance
        pred_dict: dict of str -> str. Maps the output variable name of each
                   field to predict to the tensorflow variable name in the graph.
        feed_dict: maybe(dict), tensorflow feed dict.
        data: maybe(Dataset), if no feed dict this must be set
        feed_builder: maybe(callable), if no feed dict this must be set

    Returns:
        dict of str to numpy.ndarray giving a mapping of output field name
             to predicted values
    """
    if data is None:
        return None
    if feed_dict is None and (data is None or feed_builder is None):
        raise AssertionError(
            'A feed dict or some data and a feed builder is needed to make predictions.'
        )
    if feed_dict is None:
        feed_dict = feed_builder(data, False)
    pred_names = [pred + ':0' for pred in pred_dict.values()]

    preds = sess.run(pred_names, feed_dict=feed_dict)
    pred_labels = [label for label in pred_dict.keys()]
    output_dict = {
        label: label_preds for label, label_preds in zip(pred_labels, preds)
    }
    return output_dict


def _get_validation_losses(sess, loss_names, val_feed_dict, verbose):
    loss_tensor_names = [loss_name + ':0' for loss_name in loss_names]
    losses = sess.run(loss_tensor_names, feed_dict=val_feed_dict)
    if verbose > 0:
        print('Validation Scores:')
        for loss_name, loss in zip(loss_names, losses):
            print('{}: {:0.5}'.format(loss_name, loss))
    loss_dict = dict(zip(loss_names, losses))
    return loss_dict


def _check_early_stopping(losses, scores, best_epochs, best_scores, current_epoch, n_stop):
    """Check when to stop training due to early stopping.

    The training will stop once one or more loss functions fail to improve for
    a set number of epochs.

    Args:
        losses: dict of str -> bool, keys are the loss functions, values
                are true if a greater loss value is better otherwise false.
        scores: dict of str -> float, keys are the loss function names, values
                are the current values of these losses.
        best_epochs: dict of str -> int, keys are the loss functions
        best_scores: dict of str -> float, keys are the loss functions
        current_epoch: int, the number of the current epoch.
        n_stop: int, how many epochs to wait before stopping trainng.

    Returns:
        3-tuple:
        updated_epochs: best_epochs after updating for this stopping round
        updated_scores: best_scores after updating for this stopping round
        stop_training: bool, True if training should be stopped at this point
    """
    if n_stop <= 0:
        return best_epochs, best_scores, False

    updated_epochs = {}
    updated_scores = {}
    stop_training = False
    for key in losses:
        greater_is_better = losses[key]
        new_score = scores[key]
        best_epoch = best_epochs[key]
        best_score = best_scores[key]

        if ((greater_is_better and (new_score > best_score)) or
                ((not greater_is_better) and (new_score < best_score))):
            updated_epochs[key] = current_epoch
            updated_scores[key] = new_score
        else:
            updated_epochs[key] = best_epoch
            updated_scores[key] = best_score
            if current_epoch - best_epoch > n_stop:
                stop_training = True

    return updated_epochs, updated_scores, stop_training


def _initialize_stopping(losses):
    """Initialize early stopping variables for a set of losses/scores.

    Tne losses should be given as a dict where the keys are the
    Tensorflow variable names and the values are booleans.
    The values should be true if we want to optimize for greater
    values of a variable and false if we want to optimize for lower values.

    Args:
        losses: dict of str -> bool.

    Returns:
        2-tuple giving:
        best_losses: str -> float, keys are the loss functions, values
                     are the best scores evaluated for each loss function.
        best_epochs: str -> int, keys are the loss functions, values
                     are the epochs with the best score for that loss.
    """
    best_losses = {
        loss: (2 * greater_is_better - 1) * np.inf
        for loss, greater_is_better in losses.items()
    }
    best_epochs = {loss: 0 for loss in losses.keys()}
    return best_losses, best_epochs
