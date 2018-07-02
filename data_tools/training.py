import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

from .data_types import Dataset, KFoldResults
from .util import get_number_of_outputs, get_number_of_samples, split_data_dict


def train_and_predict(
    prediction_function,
    train,
    val,
    test=None
):
    """Train on a training set and predict on a validation and optional test set.

    The prediction function has the signature:
    prediction_function[Dataset, Dataset, Dataset] -> Results

    Args:
        prediction_function: callable
        train: Dataset instance
        val: Dataset instance
        test: Dataset instance

    Returns:
        Results instance
    """
    results = prediction_function(train, val, test)
    return results


def train_and_predict_with_split(
        prediction_function,
        train,
        test=None,
        stratify_data=None,
        random_state=678,
        val_frac=0.2
):
    """Run training and predictions for a training set and optional test set.

    On the training set, validation set predictions are given after the split

    On a test set, predictions trained only on the part of the training set
    in the train part of the train/val split are used.

    The prediction function has the signature:
    prediction_function[Dataset, Dataset, Dataset] -> Results

    Args:
        prediction_function: callable
        train: Dataset instance
        test: Dataset instance
        stratify_data: array-like, data to stratify the training set before splitting
        random_state: int, the random state for splitting
        val_frac: float, the fraction of training data to use in the validation set

    Returns:
        Results instance
    """
    stratify = stratify_data is not None
    split = _select_split(random_state, val_frac, stratify)
    train_set = train.inputs.values()[0]
    idx_train, idx_val = next(split.split(train_set, stratify_data))

    train_set = Dataset(
        split_data_dict(train.inputs, idx_train),
        outputs=split_data_dict(train.outputs, idx_train),
        metadata={'description': 'training'}
    )
    val_set = Dataset(
        split_data_dict(train.inputs, idx_val),
        outputs=split_data_dict(train.outputs, idx_val),
        metadata={'description': 'validation'}
    )
    return train_and_predict(prediction_function, train_set, val_set, test=test)


def train_and_predict_with_kfold(
    prediction_function,
    train,
    test=None,
    stratify_data=None,
    random_state=456,
    shuffle=True,
    n_splits=5
):
    """Run training and predictions for a training set and optional test set.

    On the training set, out-of-fold predictions are given.

    On a test set, predictions on the model for each fold are given.

    The prediction function has the signature:
    prediction_function[Dataset, Dataset, Dataset] -> Results

    Args:
        prediction_function: callable
        train: Dataset of training data
        test: Dataset of test data
        stratify_data: array-like, data to use to stratify the splitting
        random_state: random state for the k-fold splits
        shuffle: bool, if true shuffles first
        n_splits: int, number of splits to use

    Returns:
        KFoldResults instance
    """
    stratify = stratify_data is not None
    kfold = _select_kfold(n_splits, random_state, shuffle, stratify)

    out_of_fold_results, test_results = _build_results(
        train, test, n_splits
    )

    n_train = get_number_of_samples(train)
    fold_number = np.zeros([n_train], dtype=np.int16)

    for fold, (idx_train, idx_val) in enumerate(
        kfold.split(list(train.inputs.values())[0], stratify_data)
    ):

        print('On fold {}'.format(fold))
        train_set = Dataset(
            split_data_dict(train.inputs, idx_train),
            outputs=split_data_dict(train.outputs, idx_train),
            metadata={
                'description': 'training',
                'fold': fold
            }
        )
        val_set = Dataset(
            split_data_dict(train.inputs, idx_val),
            outputs=split_data_dict(train.outputs, idx_val),
            metadata={
                'description': 'validation',
                'fold': fold
            }
        )
        fold_number[idx_val] = fold

        fold_results = prediction_function(
            train_set, val_set, test
        )

        oof_preds = fold_results.validation
        _fill_partial_results(out_of_fold_results, oof_preds, idx_val)

        if test_results is not None:
            test_preds = fold_results.test
            _fill_results(test_results, test_preds, fold)

    results = KFoldResults(
        validation=out_of_fold_results,
        test=test_results,
        train_folds=fold_number,
        metadata={
            'description': 'KFold predictions',
        }
    )

    return results


def _select_kfold(n_splits, random_state, shuffle, stratify):
    """Get a KFold object

    Args:
        n_splits: int, number of splits to use
        random_state: int, the random state
        shuffle: bool, if True, shuffles data first
        stratify: bool, if True, use stratified k-fold

    Returns:
        KFold or StratifiedKFold instance
    """
    if stratify:
        kfold = StratifiedKFold(
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle
        )
    else:
        kfold = KFold(
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle
        )
    return kfold


def _select_split(random_state, test_frac, stratify):
    """Get a shuffle split with a single split.

    Args:
        random_state: int, random state
        test_frac: float, fraction of data to include in the test/val set
        stratify: bool, whether to use a stratified split or not

    Returns:
        ShuffleSplit or StratifiedShuffleSplit instance
    """
    if stratify:
        shuffle = StratifiedShuffleSplit(
            random_state=random_state,
            n_splits=1,
            test_size=test_frac

        )
    else:
        shuffle = ShuffleSplit(
            random_state=random_state,
            n_splits=1,
            test_size=test_frac
        )
    return shuffle


def _build_results(val, test, iterations=1):
    """Build result arrays for the given data.

    Args:
        val: Dataset, the validation data
        test: maybe(Dataset), the test data
        iterations: The number of iterations to include in the validation data.

    Returns:
        2-tuple of:
        validation/out-of-fold results, dict of str to array-like
        test results, dict of str to array-like
    """
    n_val = get_number_of_samples(val)
    n_outputs = {
        key: get_number_of_outputs(output)
        for key, output in val.outputs.items()
    }

    val_results = {
        key: np.zeros([n_val, n_out])
        for key, n_out in n_outputs.items()
    }

    if test is not None:
        n_test = get_number_of_samples(test)
        if iterations > 1:
            test_results = {
                key: np.zeros([iterations, n_test, n_out])
                for key, n_out in n_outputs.items()
            }
        elif iterations == 1:
            test_results = {
                key: np.zeros([n_test, n_out])
                for key, n_out in n_outputs.items()
            }
        else:
            raise ValueError(
                'Need at least one model for the test output.'
            )

    else:
        test_results = None
    return val_results, test_results


def _fill_results(result_dict, results_to_add, iteration=0):
    """Fill the results for a given iteration.

    This does in-place modification.

    Args:
        result_dict: dict of str to array-like
        results_to_add: dict of str to array-like
        iteration: int, the iteration to add
    """
    values = list(result_dict.values())
    if not values:
        raise AssertionError(
            'Result dictionary needs to be prepared prior to filling.'
        )
    has_iterations = iteration >= 0
    if has_iterations:
        total_iterations = values[0].shape[0]
        if iteration >= total_iterations:
            raise ValueError(
                'Cannot add data for iteration {}. Data only has {} iterations.'
                .format(iteration, total_iterations)
            )
    for key, result in results_to_add.items():
        if key not in result_dict:
            raise KeyError('Key {} not found in expected results.'.format(key))
        if has_iterations:
            result_dict[key][iteration] = result
        else:
            result_dict[key][:] = result


def _fill_partial_results(result_dict, partial_results, indices):
    """Fill a set of results only at certain indices.

    All keys in the partial_results must be in the result_dict
    but the opposite need not be true.

    This does in-place modification.

    Args:
        result_dict: dict of str to array-like
        partial_results: dict of str to array-like
        indices: list-like of int, the indices in the result_dict
                 for the given partial_results.
    """
    for key, partial_result in partial_results.items():
        if key not in result_dict:
            raise KeyError(
                'Key {} not found in expected results.'
                .format(key)
            )

        result_dict[key][indices] = partial_result
