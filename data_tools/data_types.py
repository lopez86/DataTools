"""Defines basic data types."""


class Dataset:
    """Base Dataset class."""
    def __init__(self, inputs, outputs=None, metadata=None):
        """Create a basic dataset object.

        This contains 3 maps giving input arrays,
        output arrays, and any additional metadata the user
        wants to include.

        The basic form of the input and output arrays should be
        something like:

        input = {
            'x1': np.ndarray(),
            'x2': np.ndarray(),
            ...
        }

        Other array-like types can be used in many cases too.
        These include scipy sparse arrays, numpy matrices,
        pandas DataFrames, and even lists.

        Args:
            inputs: dict of str -> array-like
            outputs: dict of str -> array-like
            metadata: dict
        """
        self._inputs = inputs
        self._outputs = outputs
        self._metadata = metadata

    @property
    def inputs(self):
        """Get the input dict."""
        return self._inputs

    @property
    def outputs(self):
        """Get the output dict."""
        return self._outputs

    @property
    def metadata(self):
        """Get the metadata dict."""
        return self._metadata


class BatchData(Dataset):
    """Dataset to represent a training batch"""
    def __init__(
            self,
            inputs,
            outputs=None,
            epoch=0,
            epoch_done=False,
            batch=0,
            metadata=None
    ):
        """Create an object to represent a training batch.

        This is more or less the same as the regular Dataset except it also
        has information about the current location in training.

        Args:
            inputs: dict of str to array-like
            outputs: dict of str to array-like
            epoch: int, the current epoch
            epoch_done: bool, True if the last batch of the current epoch
            batch: int, the index of the current batch in the current epoch
            metadata: dict
        """
        super(BatchData, self).__init__(inputs, outputs, metadata)
        self._batch = batch
        self._epoch = epoch
        self._epoch_done = epoch_done

    @property
    def batch(self):
        """Get the index of the current batch, resetting on new epochs."""
        return self._batch

    @property
    def epoch(self):
        """Get the index of the current epoch."""
        return self._epoch

    @property
    def epoch_done(self):
        """Get whether or not this is the end of the current epoch."""
        return self._epoch_done


class Results:
    """Container for prediction results."""
    def __init__(self, validation=None, test=None, metadata=None):
        """Create an object to hold predictions.

        Args:
            validation: dict of str to array-like
            test: dict of str to array-like
            metadata: dict
        """
        self._validation = validation
        self._test = test
        self._metadata = metadata

    @property
    def validation(self):
        """Get the predictions on the validation set."""
        return self._validation

    @property
    def test(self):
        """Get the predictions on the test set."""
        return self._test

    @property
    def metadata(self):
        """Get any other metadata."""
        return self._metadata


class KFoldResults(Results):
    """Hold results from a KFold cross validation."""
    def __init__(self, validation=None, test=None, train_folds=None, metadata=None):
        """Create an object to hold KFold CV output.

        Mostly the same as regular results except this holds an array
        to give the fold at which each training sample was in the validation set.

        Note that the validation data here should be the out of fold results
        for the entire training set, and the test set will typically be the
        predictions for each fold.

        Args:
            validation: dict of str -> array-like
            test: dict of str -> array-like
            train_folds: Array-like of ints, the folds for each train sample
            metadata: dict, any other metadata
        """
        super(KFoldResults, self).__init__(validation, test, metadata)
        self._folds = train_folds

    @property
    def train_folds(self):
        """Get where each of the training samples was in the out of fold sample."""
        return self._folds
