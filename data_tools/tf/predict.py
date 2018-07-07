import funcy
import numpy as np
import tensorflow as tf

from .feed_dict import simple_feed_builder
from ..batches import generate_batches
from ..data_types import Results
from ..util import get_number_of_samples

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
    predict_batch_generator=None,
    n_stop=-1,
    verbose=1,
    losses=None,
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
        predict_batch_generator: maybe(callable), an alternative batch generator
                                 for predictions on the test & validation sets
        n_stop: int, number of epochs with no improvement before stopping.
                If 0 or negative, the stopping is disabled.
        verbose: int, verbosity level
        losses: list of Loss tuples
        train_str: str, the training operation for the graph
        pred_dict: dict of str -> str, gives the mapping of output variable name to
                   tensorflow graph variable name.


    Returns:
        Results instance with predictions on any validation or test data.
    """
    best_scores, best_epochs = _initialize_stopping(losses)
    if predict_batch_generator is None:
        predict_batch_generator = batch_generator

    train, val, test, features = preprocessor(
        train_data, val_data, test_data
    )
    model = model_builder(train, val, test, features)

    with tf.Session(graph=model) as sess:
        tf.global_variables_initializer().run()

        for batch in batch_generator(train):
            train_feed_dict = feed_builder(batch, True)

            sess.run(train_str, feed_dict=train_feed_dict)
            if batch.epoch_done and losses is not None and val is not None:

                if verbose == 0 and n_stop <= 0:
                    continue

                scores = _get_validation_losses(
                    sess, val, losses, predict_batch_generator, feed_builder, verbose
                )

                best_epochs, best_scores, stop_training = _check_early_stopping(
                    losses, scores, best_epochs, best_scores, batch.epoch, n_stop
                )

                if stop_training:
                    if verbose > 0:
                        print('Stopping training at epoch {}'.format(batch.epoch))
                    break

        if verbose > 0 and val is not None:
            print('Making validation set predictions')
        val_preds = _run_predictions(
            sess,
            pred_dict,
            data=val,
            batch_generator=predict_batch_generator,
            feed_builder=feed_builder
        )
        if verbose > 0 and test is not None:
            print('Making test set predictions')
        test_preds = _run_predictions(
            sess,
            pred_dict,
            data=test,
            batch_generator=predict_batch_generator,
            feed_builder=feed_builder
        )

        results = Results(validation=val_preds, test=test_preds)
        return results


def _run_predictions(sess, pred_dict, data, batch_generator, feed_builder):
    """With the current active session, make predictions from a given dataset.

    A feed dict may be passed directly into this function but otherwise, a Dataset
    and a function to build a feed dict must be used instead.

    Args:
        sess: tensorflow.Session instance
        pred_dict: dict of str -> str, gives the mapping of output variable name to
                   tensorflow graph variable name.
        data: Dataset
        batch_generator: callable
        feed_builder: callable

    Returns:
        dict of str to numpy.ndarray giving a mapping of output field name
             to predicted values
    """
    if data is None:
        return None

    n_total = get_number_of_samples(data)
    # Get the tensor names to be sent to tensorflow
    pred_names = [pred + ':0' for pred in pred_dict.values()]
    # Get the correct output shapes
    pred_shapes = [
        tf.get_default_graph().get_tensor_by_name(pred_name).get_shape().as_list()
        for pred_name in pred_names
    ]
    for pred_shape in pred_shapes:
        pred_shape[0] = n_total
    # Get the output labels and build the output arrays
    pred_labels = [label for label in pred_dict.keys()]
    output_dict = {
        label: np.zeros(shape, dtype=np.float32)
        for label, shape in zip(pred_labels, pred_shapes)
    }

    # Loop through batches
    for batch in batch_generator(data):
        feed_dict = feed_builder(batch, False)
        preds = sess.run(pred_names, feed_dict=feed_dict)
        for label, pred in zip(pred_labels, preds):
            output_dict[label][batch.index] = pred
        if batch.epoch_done:
            break

    return output_dict


def _get_validation_losses(
        sess,
        data,
        losses,
        batch_generator,
        feed_builder,
        verbose):
    """Calculate losses for a given dataset.

    Args:
        sess: tensorflow.Session
        data: Dataset instance
        losses: list of Loss tuples
        batch_generator: callable
        feed_builder: callable
        verbose: int

    Returns:
        dict of str -> float
    """
    loss_tensor_names = funcy.ldistinct([loss.tensor for loss in losses])
    tensor_dict = {name: name for name in loss_tensor_names}
    result_dict = _run_predictions(
        sess, tensor_dict, data, batch_generator, feed_builder
    )

    loss_dict = {
        loss.name: loss.function(
            data.outputs[loss.field], result_dict[loss.tensor]
        )
        for loss in losses
    }

    if verbose > 0:
        print('Validation Scores:')
        for loss_name, loss in loss_dict.items():
            print('{}: {:0.5}'.format(loss_name, loss))
    return loss_dict


def _check_early_stopping(losses, scores, best_epochs, best_scores, current_epoch, n_stop):
    """Check when to stop training due to early stopping.

    The training will stop once one or more loss functions fail to improve for
    a set number of epochs.

    Args:
        losses: list of Loss tuples
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
    for loss in losses:
        loss_name = loss.name
        greater_is_better = loss.greater_is_better
        new_score = scores[loss_name]
        best_epoch = best_epochs[loss_name]
        best_score = best_scores[loss_name]

        if ((greater_is_better and (new_score > best_score)) or
                ((not greater_is_better) and (new_score < best_score))):
            updated_epochs[loss_name] = current_epoch
            updated_scores[loss_name] = new_score
        else:
            updated_epochs[loss_name] = best_epoch
            updated_scores[loss_name] = best_score
            if current_epoch - best_epoch > n_stop:
                stop_training = True

    return updated_epochs, updated_scores, stop_training


def _initialize_stopping(losses):
    """Initialize early stopping variables for a set of losses/scores.

    Args:
        losses: list of Loss tuples.

    Returns:
        2-tuple giving:
        best_losses: str -> float, keys are the loss functions, values
                     are the best scores evaluated for each loss function.
        best_epochs: str -> int, keys are the loss functions, values
                     are the epochs with the best score for that loss.
    """
    best_losses = {
        loss.name: (2 * loss.greater_is_better - 1) * np.inf
        for loss in losses
    }
    best_epochs = {loss.name: 0 for loss in losses}
    return best_losses, best_epochs
